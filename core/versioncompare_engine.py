# core/engine_flow4.py
"""
Flow 4: Version Compare (Smart Redlines)
Compares two document versions and explains what changed and why it matters.
"""

import json
import base64
import io
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_LEFT, TA_CENTER

import difflib
import re

import config
from .document_processor import DocumentProcessor
from .schemas import VersionCompareResponse, ChangeLogEntry


class ChangeCategory(str, Enum):
    """Categories of changes for classification"""
    ECONOMICS = "economics"  # Payment terms, pricing, penalties
    RISK = "risk"  # Liability, indemnification, warranties
    TIMING = "timing"  # Deadlines, notice periods, term length
    SCOPE = "scope"  # Deliverables, obligations, definitions
    GOVERNANCE = "governance"  # Decision rights, approvals, dispute resolution
    COMPLIANCE = "compliance"  # Legal requirements, certifications
    OTHER = "other"


class VersionCompareEngine:
    """
    Performs intelligent document comparison with legal context awareness.
    """

    def __init__(self, file_v1, file_v2, topic_focus: Optional[List[str]] = None):
        """
        Args:
            file_v1: First version (typically older)
            file_v2: Second version (typically newer)
            topic_focus: Optional list of topics to focus analysis on
        """
        self.processor_v1 = DocumentProcessor(file_v1)
        self.processor_v2 = DocumentProcessor(file_v2)
        self.topic_focus = topic_focus or ["all"]
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def compare(self) -> VersionCompareResponse:
        """
        Execute the full comparison pipeline.
        Returns structured comparison results with change log and PDF.
        """
        print("Starting version comparison...")
        
        # 1. Extract text from both versions
        text_v1, quality_flags_v1 = self.processor_v1.process()
        text_v2, quality_flags_v2 = self.processor_v2.process()
        
        quality_flags = list(set(quality_flags_v1 + quality_flags_v2))
        
        # 2. Perform structural diff to identify changed sections
        structural_changes = self._compute_structural_diff(text_v1, text_v2)
        
        # 3. Use LLM to analyze changes and explain impact
        change_log = self._analyze_changes_with_llm(
            text_v1, text_v2, structural_changes
        )
        
        # 4. Generate comparison PDF report
        pdf_base64 = self._create_comparison_pdf(change_log, text_v1, text_v2)
        
        print(f"Comparison complete. Found {len(change_log)} significant changes.")
        
        return VersionCompareResponse(
            change_log=change_log,
            summary_pdf_base64=pdf_base64,
            quality_flags=quality_flags,
            total_changes=len(change_log)
        )

    def _compute_structural_diff(self, text_v1: str, text_v2: str) -> List[Dict]:
        """
        Compute line-by-line diff and identify changed sections.
        Returns structured list of changes with context.
        """
        lines_v1 = text_v1.splitlines()
        lines_v2 = text_v2.splitlines()
        
        # Use difflib for intelligent line matching
        differ = difflib.Differ()
        diff = list(differ.compare(lines_v1, lines_v2))
        
        changes = []
        current_change = None
        context_before = []
        context_after = []
        
        for i, line in enumerate(diff):
            prefix = line[:2]
            content = line[2:].strip()
            
            if prefix == '- ':  # Deleted line
                if current_change and current_change['type'] != 'deletion':
                    # Save previous change
                    changes.append(current_change)
                    current_change = None
                
                if not current_change:
                    current_change = {
                        'type': 'deletion',
                        'old_text': content,
                        'new_text': '',
                        'context_before': context_before[-3:],  # Last 3 lines
                        'line_number_v1': self._estimate_line_number(lines_v1, content)
                    }
                else:
                    current_change['old_text'] += '\n' + content
            
            elif prefix == '+ ':  # Added line
                if current_change and current_change['type'] == 'deletion':
                    # This is a modification (deletion + addition)
                    current_change['type'] = 'modification'
                    current_change['new_text'] = content
                    current_change['line_number_v2'] = self._estimate_line_number(lines_v2, content)
                elif current_change and current_change['type'] == 'modification':
                    current_change['new_text'] += '\n' + content
                else:
                    if current_change:
                        changes.append(current_change)
                    current_change = {
                        'type': 'addition',
                        'old_text': '',
                        'new_text': content,
                        'context_before': context_before[-3:],
                        'line_number_v2': self._estimate_line_number(lines_v2, content)
                    }
            
            elif prefix == '  ':  # Unchanged line
                if current_change:
                    current_change['context_after'] = [content]
                    changes.append(current_change)
                    current_change = None
                context_before.append(content)
        
        # Save final change if exists
        if current_change:
            changes.append(current_change)
        
        # Filter out trivial changes (whitespace only, too short)
        significant_changes = [
            c for c in changes 
            if len(c.get('old_text', '') + c.get('new_text', '').strip()) > 10
        ]
        
        return significant_changes

    def _estimate_line_number(self, lines: List[str], content: str) -> Optional[int]:
        """Estimate line number by searching for content match."""
        for i, line in enumerate(lines, 1):
            if content in line:
                return i
        return None

    def _analyze_changes_with_llm(
        self, 
        text_v1: str, 
        text_v2: str, 
        structural_changes: List[Dict]
    ) -> List[ChangeLogEntry]:
        """
        Use LLM to analyze each change and explain its legal/business impact.
        """
        if not structural_changes:
            return []
        
        # Prepare change summary for LLM
        changes_summary = self._format_changes_for_llm(structural_changes[:50])  # Limit to 50 changes
        
        system_prompt = """
You are LUKE, an expert legal analyst specializing in contract review and risk assessment.

Your task is to analyze changes between two document versions and explain:
1. **What changed** (concise summary)
2. **Why it matters** (legal/business impact)
3. **Risk level** (critical, high, medium, low)
4. **Category** (economics, risk, timing, scope, governance, compliance, other)
5. **Negotiation points** (if applicable)

CRITICAL RULES:
- Focus ONLY on substantive changes (ignore formatting, typos, minor wording)
- For each change, provide the EXACT text from both versions
- Be specific about impact (e.g., "increases liability by $X" not "changes liability")
- If topic_focus is specified, prioritize those areas but don't ignore critical changes elsewhere
- Use legal terminology but remain clear and actionable

Respond in JSON format:
{
    "changes": [
        {
            "change_type": "addition|deletion|modification",
            "section_title": "...",
            "old_text": "...",
            "new_text": "...",
            "summary": "Brief description of what changed",
            "impact_explanation": "Detailed explanation of legal/business impact",
            "impact_level": "critical|high|medium|low",
            "category": "economics|risk|timing|scope|governance|compliance|other",
            "negotiation_points": ["point 1", "point 2"],
            "page_v1": <int or null>,
            "page_v2": <int or null>
        }
    ]
}
"""

        topics_instruction = f"\nFocus areas: {', '.join(self.topic_focus)}" if "all" not in self.topic_focus else ""
        
        user_prompt = f"""
Analyze the following changes between Version 1 and Version 2 of a legal document.
{topics_instruction}

STRUCTURAL CHANGES DETECTED:
---
{changes_summary}
---

FULL VERSION 1 (for context):
---
{text_v1[:80000]}
---

FULL VERSION 2 (for context):
---
{text_v2[:80000]}
---

Provide a comprehensive analysis focusing on substantive changes that affect rights, obligations, risks, or economics.
"""

        try:
            response = self.client.chat.completions.create(
                model=config.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert to ChangeLogEntry objects
            change_log = []
            for change_data in result.get("changes", []):
                try:
                    change_log.append(ChangeLogEntry(
                        change_type=change_data.get("change_type", "modification"),
                        section_title=change_data.get("section_title", "Unspecified Section"),
                        old_text=change_data.get("old_text", ""),
                        new_text=change_data.get("new_text", ""),
                        summary=change_data.get("summary", ""),
                        impact_explanation=change_data.get("impact_explanation", ""),
                        impact_level=change_data.get("impact_level", "medium"),
                        category=change_data.get("category", "other"),
                        negotiation_points=change_data.get("negotiation_points", []),
                        page_v1=change_data.get("page_v1"),
                        page_v2=change_data.get("page_v2")
                    ))
                except Exception as e:
                    print(f"Warning: Failed to parse change entry: {e}")
                    continue
            
            return change_log
            
        except Exception as e:
            print(f"Error during LLM analysis: {e}")
            return []

    def _format_changes_for_llm(self, changes: List[Dict]) -> str:
        """Format structural changes into readable text for LLM."""
        formatted = []
        for i, change in enumerate(changes, 1):
            formatted.append(f"\n--- Change {i} ---")
            formatted.append(f"Type: {change['type']}")
            if change.get('old_text'):
                formatted.append(f"OLD: {change['old_text'][:300]}")
            if change.get('new_text'):
                formatted.append(f"NEW: {change['new_text'][:300]}")
            if change.get('context_before'):
                formatted.append(f"Context: {' '.join(change['context_before'][-2:])}")
        
        return '\n'.join(formatted)

    def _create_comparison_pdf(
        self, 
        change_log: List[ChangeLogEntry],
        text_v1: str,
        text_v2: str
    ) -> str:
        """
        Generate a professional comparison PDF report.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=10,
            spaceBefore=15
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            leading=14
        )
        
        # --- HEADER ---
        story.append(Paragraph("Document Version Comparison Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # --- EXECUTIVE SUMMARY ---
        story.append(Paragraph("Executive Summary", section_style))
        
        summary_data = self._generate_summary_stats(change_log)
        summary_text = f"""
        <b>Total Changes:</b> {summary_data['total']}<br/>
        <b>Critical Changes:</b> {summary_data['critical']}<br/>
        <b>High Impact:</b> {summary_data['high']}<br/>
        <b>Medium Impact:</b> {summary_data['medium']}<br/>
        <b>Low Impact:</b> {summary_data['low']}<br/>
        <br/>
        <b>Most Affected Categories:</b> {', '.join(summary_data['top_categories'])}
        """
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 0.3*inch))
        
        # --- DETAILED CHANGE LOG ---
        story.append(Paragraph("Detailed Change Log", section_style))
        
        # Sort by impact level
        impact_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_changes = sorted(
            change_log, 
            key=lambda x: impact_order.get(x.impact_level, 4)
        )
        
        for i, change in enumerate(sorted_changes, 1):
            story.extend(self._format_change_for_pdf(change, i, styles))
            story.append(Spacer(1, 0.2*inch))
            
            # Page break every 3 changes for readability
            if i % 3 == 0 and i < len(sorted_changes):
                story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _generate_summary_stats(self, change_log: List[ChangeLogEntry]) -> Dict:
        """Generate summary statistics for the report."""
        from collections import Counter
        
        impact_counts = Counter(c.impact_level for c in change_log)
        category_counts = Counter(c.category for c in change_log)
        
        return {
            'total': len(change_log),
            'critical': impact_counts.get('critical', 0),
            'high': impact_counts.get('high', 0),
            'medium': impact_counts.get('medium', 0),
            'low': impact_counts.get('low', 0),
            'top_categories': [cat for cat, _ in category_counts.most_common(3)]
        }

    def _format_change_for_pdf(self, change: ChangeLogEntry, index: int, styles) -> List:
        """Format a single change entry for PDF inclusion."""
        elements = []
        
        # Impact color coding
        impact_colors = {
            "critical": colors.HexColor('#d32f2f'),
            "high": colors.HexColor('#f57c00'),
            "medium": colors.HexColor('#fbc02d'),
            "low": colors.HexColor('#388e3c')
        }
        
        impact_color = impact_colors.get(change.impact_level, colors.grey)
        
        # Change header
        header_style = ParagraphStyle(
            'ChangeHeader',
            parent=styles['Heading3'],
            fontSize=11,
            textColor=impact_color,
            spaceAfter=8
        )
        
        header_text = f"{index}. {change.section_title} [{change.impact_level.upper()}]"
        elements.append(Paragraph(header_text, header_style))
        
        # Summary
        elements.append(Paragraph(f"<b>Summary:</b> {change.summary}", styles['BodyText']))
        
        # Impact explanation
        elements.append(Paragraph(
            f"<b>Impact:</b> {change.impact_explanation}", 
            styles['BodyText']
        ))
        
        # Old vs New text table
        if change.old_text or change.new_text:
            table_data = [
                [Paragraph("<b>Version 1</b>", styles['BodyText']), 
                 Paragraph("<b>Version 2</b>", styles['BodyText'])],
                [Paragraph(change.old_text[:500] or "<i>Not present</i>", styles['BodyText']),
                 Paragraph(change.new_text[:500] or "<i>Removed</i>", styles['BodyText'])]
            ]
            
            change_table = Table(table_data, colWidths=[3.25*inch, 3.25*inch])
            change_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTSIZE', (0, 0), (-1, -1), 9)
            ]))
            
            elements.append(change_table)
        
        # Negotiation points
        if change.negotiation_points:
            neg_text = "<b>Negotiation Points:</b><br/>" + "<br/>".join(
                f"• {point}" for point in change.negotiation_points
            )
            elements.append(Paragraph(neg_text, styles['BodyText']))
        
        return elements