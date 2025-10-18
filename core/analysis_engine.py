# core/engine_flow2.py
import io
import json
import base64
from typing import List, Optional

from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

import config
from .document_processor import DocumentProcessor
from .schemas import StructuredData, Finding


class AnalysisEngine:
    """Orchestrates AI tasks for document analysis and report generation."""

    def __init__(self, file, topics: Optional[List[str]] = None):
        self.processor, quality_flags = DocumentProcessor(file)
        self.full_text,  = self.processor.process()
        self.topics = topics or []
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def analyze(self) -> (str, List[Finding], StructuredData, str):
        """
        Performs the full analysis pipeline.
        Returns a tuple of (summary, findings, structured_data, pdf_base64).
        """
        # Truncate text if too long to avoid excessive API costs/limits
        max_chars = 100000
        truncated_text = self.full_text[:max_chars]

        analysis_json = self._run_analysis_llm_call(truncated_text)

        # Extract outputs with defaults
        executive_summary = analysis_json.get("executive_summary", "Summary could not be generated.")
        findings_data = analysis_json.get("findings", [])
        structured_data_map = analysis_json.get("structured_data", {})

        # Convert raw dicts to Pydantic models for validation
        findings = [Finding(**f) for f in findings_data]

        # --- SANITIZE STRUCTURED DATA TO AVOID NONE ---
        structured_data_map.setdefault("parties", [])
        structured_data_map["parties"] = [
            str(p) if p is not None else "Unknown Party"
            for p in structured_data_map["parties"]
        ]

        # Ensure other optional fields are strings or None
        structured_data_map["effective_date"] = structured_data_map.get("effective_date") or None
        structured_data_map["termination_date"] = structured_data_map.get("termination_date") or None
        structured_data_map["key_deadlines"] = structured_data_map.get("key_deadlines") or {}
        structured_data_map["monetary_amounts"] = [
            str(m) for m in structured_data_map.get("monetary_amounts", []) if m is not None
        ]

        structured_data = StructuredData(**structured_data_map)

        # Generate the one-page PDF report
        pdf_bytes = self._create_one_pager_pdf(executive_summary, findings, structured_data)
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        return executive_summary, findings, structured_data, pdf_base64

    def _run_analysis_llm_call(self, text: str) -> dict:
        """Runs a single, powerful LLM call to get all analysis in one shot."""
        topics_str = ", ".join(self.topics) if self.topics else "key clauses, risks, and obligations"

        system_prompt = """
        You are an expert legal analyst. Your task is to review a legal document and produce a structured JSON output.
        You must adhere strictly to the provided JSON format.
        For each finding, you must provide an exact 'quote' from the document that supports your summary.
        Extract structured data accurately. If a piece of data is not found, use a null value.
        """

        user_prompt = f"""
        Please analyze the following document.
        Produce a one-page executive summary (in expert tone).
        Identify and summarize findings related to these specific topics: {topics_str}.
        Extract the specified structured data.

        DOCUMENT TEXT:
        ---
        {text}
        ---

        Please provide your output in the following JSON format:
        {{
            "executive_summary": "...",
            "findings": [
                {{
                    "topic": "...",
                    "summary": "...",
                    "page_number": <integer>,
                    "quote": "..."
                }}
            ],
            "structured_data": {{
                "parties": ["...", "..."],
                "effective_date": "...",
                "termination_date": "...",
                "key_deadlines": {{ "deadline_name": "YYYY-MM-DD" }},
                "monetary_amounts": ["..."]
            }}
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=config.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error during LLM analysis call: {e}")
            return {}

    def _create_one_pager_pdf(self, summary: str, findings: List[Finding], s_data: StructuredData) -> bytes:
        """Generates a one-page PDF report in memory."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Document Analysis Report", styles['h1']))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Executive Summary", styles['h2']))
        story.append(Paragraph(summary.replace('\n', '<br/>'), styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Key Findings", styles['h2']))
        for find in findings:
            story.append(Paragraph(f"<b>Topic:</b> {find.topic}", styles['h3']))
            story.append(Paragraph(f"<b>Summary:</b> {find.summary}", styles['BodyText']))
            story.append(Paragraph(f"<i><b>Quote (Page {find.page_number}):</b> \"{find.quote}\"</i>", styles['Italic']))
            story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph("Structured Data", styles['h2']))
        s_text = f"""
        <b>Parties:</b> {', '.join(s_data.parties) if s_data.parties else 'N/A'}<br/>
        <b>Effective Date:</b> {s_data.effective_date or 'N/A'}<br/>
        <b>Termination Date:</b> {s_data.termination_date or 'N/A'}<br/>
        """
        story.append(Paragraph(s_text, styles['BodyText']))

        doc.build(story)
        return buffer.getvalue()
