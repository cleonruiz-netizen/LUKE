import io, json, base64, os
from typing import List
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from pydantic import ValidationError
from supabase import create_client, Client

import config
from .document_processor import DocumentProcessor
from .schemas import ClauseAuditItem, ClauseAuditResponse


class ClauseAuditEngine:
    """Performs clause presence/deviation audit on a contract."""

    def __init__(self, file, checklist: List[str]):
        self.processor = DocumentProcessor(file)
        self.checklist = [c.lower() for c in checklist]
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

        # --- Supabase Setup ---
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

        if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET]):
            raise ValueError(
                "Missing Supabase credentials. Please set SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET in .env"
            )

        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

        # --- Load clause baselines JSON from Supabase path specified in config ---
        self.baselines = self._load_clause_baselines_from_supabase()



    def _load_clause_baselines_from_supabase(self) -> dict:
        """
        Downloads the clause baselines JSON file from Supabase.
        The file path (e.g., 'baselines/clause_baselines.json') is defined in config.CLAUSE_BASELINES_PATH.
        """
        try:
            data = self.supabase.storage.from_(os.getenv("SUPABASE_BUCKET")).download(
                config.CLAUSE_BASELINES_PATH
            )
            print(f"☁️ Loaded clause baselines from Supabase: {config.CLAUSE_BASELINES_PATH}")
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            print(f"⚠ Could not load baselines from Supabase: {e}")
            # fallback to local file if available
            try:
                with open(config.CLAUSE_BASELINES_PATH, "r", encoding="utf-8") as f:
                    print("📄 Fallback: loaded local clause_baselines.json")
                    return json.load(f)
            except Exception:
                print("❌ No baseline file available.")
                return {}



    def audit(self) -> ClauseAuditResponse:
        """Full audit pipeline."""
        full_text, quality_flags = self.processor.process()
        findings = self._run_audit_llm(full_text)

        # Strict JSON validation using Pydantic
        items = []
        for f in findings.get("audited_clauses", []):
            try:
                items.append(ClauseAuditItem(**f))
            except ValidationError:
                items.append(
                    ClauseAuditItem(
                        clause_name=f.get("clause_name", "Unknown"),
                        status=f.get("status", "MISSING"),
                        summary="Invalid or incomplete LLM output",
                        page_number=None,
                        quote="",
                    )
                )

        pdf_b64 = self._create_pdf(items)
        return ClauseAuditResponse(
            audit_table=items,
            one_pager_pdf_base64=pdf_b64,
            quality_flags=quality_flags,
        )



    def _run_audit_llm(self, text: str) -> dict:
        """Runs the clause audit process with LLM and returns JSON."""

        system_prompt = """
        You are LUKE, an expert legal auditor and compliance analyst.

        Your task:
        1. Determine the type of document (e.g., Service Agreement, NDA, Employment Contract, etc.).
        2. You are given a list of baseline clauses for reference.
        3. Audit only the clauses specified by the user. The user’s clause names may differ from the baseline names, so match them as best you can.
        4. For each clause you audit, classify as:
            • PRESENT → clause exists and fulfills purpose
            • DEVIATES → clause exists but with differences/reduced protection
            • MISSING → clause does not exist
        5. Provide for each clause:
            - clause_name (user’s name or closest match)
            - status ("PRESENT", "DEVIATES", "MISSING")
            - summary
            - page_number (if identifiable)
            - quote (supporting text)
        6. Focus on the intent, not exact wording. Be concise, factual, professional.
        7. Respond ONLY in JSON format.
        """

        user_prompt = f"""
        Document text (truncated to 80k chars):
        ---
        {text[:80000]}
        ---
        Baseline clauses (for reference only):

        {json.dumps(self.baselines, indent=2)}

        User-specified clauses to audit (may not exactly match baseline names):
        {', '.join(self.checklist)}

        Analyze the document, infer its type, and audit only the user-specified clauses.
        Map them to baseline clauses if needed, but only include those requested by the user.
        """

        response = self.client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"document_type": "Unknown", "audited_clauses": []}



    def _create_pdf(
        self,
        items: List["ClauseAuditItem"],
        document_type: str = "Unknown",
        quality_flags: List[str] = [],
    ) -> str:
        """Generates a professional one-page PDF audit report with color-coded statuses and wrapped summaries."""

        def color_to_hex(color):
            r = int(color.red * 255)
            g = int(color.green * 255)
            b = int(color.blue * 255)
            return f"#{r:02X}{g:02X}{b:02X}"

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch
        )
        styles = getSampleStyleSheet()
        header_style = styles["Heading1"]
        body_style = ParagraphStyle("body", fontSize=10, leading=12)
        cell_style = ParagraphStyle("cell", fontSize=9, leading=11, wordWrap="CJK")

        story = []
        story.append(Paragraph("Clause Audit Report", header_style))
        story.append(Spacer(1, 0.1 * inch))

        if quality_flags:
            story.append(
                Paragraph(f"<b>Quality Flags:</b> {', '.join(quality_flags)}", body_style)
            )
        story.append(Spacer(1, 0.2 * inch))

        data = [
            [
                Paragraph("Clause", styles["Heading4"]),
                Paragraph("Status", styles["Heading4"]),
                Paragraph("Summary", styles["Heading4"]),
                Paragraph("Quote", styles["Heading4"]),
                Paragraph("Page", styles["Heading4"]),
            ]
        ]

        status_colors = {
            "PRESENT": colors.green,
            "DEVIATES": colors.orange,
            "MISSING": colors.red,
        }

        for it in items:
            status_color = status_colors.get(it.status, colors.black)
            hex_color = color_to_hex(status_color)

            data.append(
                [
                    Paragraph(it.clause_name, cell_style),
                    Paragraph(f'<font color="{hex_color}">{it.status}</font>', cell_style),
                    Paragraph(it.summary, cell_style),
                    Paragraph(it.quote or "-", cell_style),
                    Paragraph(str(it.page_number or "—"), cell_style),
                ]
            )

        table = Table(
            data,
            colWidths=[1.5 * inch, 0.9 * inch, 3 * inch, 2 * inch, 0.6 * inch],
            repeatRows=1,
        )
        table_style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("ALIGN", (1, 1), (1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )

        def lighten_color(color, factor=0.8):
            r = color.red + (1 - color.red) * factor
            g = color.green + (1 - color.green) * factor
            b = color.blue + (1 - color.blue) * factor
            return colors.Color(r, g, b)

        for row_idx, it in enumerate(items, start=1):
            bg_color = status_colors.get(it.status, colors.white)
            bg_color_lite = lighten_color(bg_color, 0.8)
            table_style.add("BACKGROUND", (1, row_idx), (1, row_idx), bg_color_lite)

        table.setStyle(table_style)
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))
        story.append(
            Paragraph(
                "<i>Note: Status colors → Green = PRESENT, Orange = DEVIATES, Red = MISSING</i>",
                styles["Italic"],
            )
        )

        doc.build(story)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
