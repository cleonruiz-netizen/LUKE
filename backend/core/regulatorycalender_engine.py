from datetime import datetime, date
from dateutil.relativedelta import relativedelta 
import base64
from ics import Calendar, Event
from core.document_processor import DocumentProcessor
from core.schemas import RegulatoryCalendarResponse, CalendarItem
from fastapi import UploadFile, HTTPException
import json
from openai import OpenAI
from config import OPENAI_API_KEY, GENERATION_MODEL
from ics.grammar.parse import ContentLine 
from typing import List, Optional

class RegulatoryCalendarEngine:
    """Extracts deadlines and notice periods and produces calendar items."""

    def __init__(self, file):
        self.processor = DocumentProcessor(file)
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract(self)->RegulatoryCalendarResponse:
        """Full extraction pipeline."""
        full_text, quality_flags = self.processor.process()
        
        if "low_ocr_confidence" in quality_flags and not full_text:
            return RegulatoryCalendarResponse(calendar_items=[], ics_file_base64="", quality_flags=quality_flags)

        extracted_items_json = self._extract_deadlines_with_llm(full_text)
        
        deadlines_data = extracted_items_json.get("deadlines", [])
        calendar_items = [CalendarItem(**item) for item in deadlines_data]
        ics_base64 = self._create_ics_file(deadlines_data)
        
        return RegulatoryCalendarResponse(
            calendar_items=calendar_items,
            ics_file_base64=ics_base64,
            quality_flags=quality_flags
        )

    def _extract_deadlines_with_llm(self, text: str) -> dict:
        """Uses an LLM to find and structure all deadlines, providing the current date for context."""
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt = (
            "You are an expert paralegal specializing in compliance. Your task is to meticulously extract all deadlines, "
            "notice periods, and recurring obligations from a legal document. You must structure the date and recurrence "
            "information precisely."
        )
        user_prompt = f"""
        Today's date is {current_date_str}. Use this for context when interpreting relative dates.

        Analyze the document and extract all deadlines. For each deadline:
        1. `is_plannable`: Set to `true` ONLY if the deadline is an absolute date (e.g., "2025-12-31") or a clearly defined recurring event (e.g., "annually on March 31"). For vague, relative deadlines (e.g., "30 days after a notice is sent"), set this to `false`.
        2. `event_date`: If `is_plannable` is true and it's an absolute date, resolve it to "YYYY-MM-DD". Otherwise, describe the rule textually (e.g., "Annual, by March 31", "Within 15 days of a triggering event").
        3. `recurrence`: If the event is recurring, provide a structured object. Otherwise, use null.
           - `frequency`: "YEARLY", "MONTHLY", "WEEKLY"
           - `by_month`: <integer>
           - `by_month_day`: <integer>
        4. Provide `title`, `category`, `responsible_party`, and `citation`.

        DOCUMENT TEXT: --- {text[:80000]} ---
        Respond ONLY in the specified JSON format.
        """
        try:
            response = self.client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"AI service failed to extract deadlines: {e}")

    def _translate_recurrence_to_rrule(self, recurrence: dict) -> str | None:
        """Uses a targeted LLM call to convert a structured recurrence object into an RFC 5545 RRULE string."""
        if not recurrence or not recurrence.get("frequency"):
            return None
        system_prompt = (
            "You are an expert RFC 5545 calendar rule generator. Convert the provided JSON object "
            "into a valid RRULE string. Do not include the 'RRULE:' prefix. If invalid, respond with an empty string."
        )
        user_prompt = f"Convert to RRULE: {json.dumps(recurrence)}"
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0
            )
            rrule_string = response.choices[0].message.content.strip()
            return rrule_string if "FREQ" in rrule_string else None
        except Exception:
            return None

    def _create_ics_file(self, raw_deadlines: List[dict]) -> str:
        """Generates a Base64 encoded .ics file ONLY for plannable events."""
        cal = Calendar()
        
        for item in raw_deadlines:
            if not item.get("is_plannable", False):
                continue

            event = Event()
            event.name = f"[{item.get('category', 'Task')}] {item.get('title', 'Untitled Event')}"
            event.description = (
                f"Rule: {item.get('event_date', 'N/A')}\n"
                f"Responsible Party: {item.get('responsible_party', 'N/A')}\n\n"
                f"Citation:\n\"{item.get('citation', 'N/A')}\""
            )

            is_recurring = False
            recurrence_data = item.get("recurrence")
            if recurrence_data:
                rrule_string = self._translate_recurrence_to_rrule(recurrence_data)
                if rrule_string:
                    today = date.today()
                    by_month = recurrence_data.get("by_month", today.month)
                    by_day = recurrence_data.get("by_month_day", today.day)
                    try:
                        event_date_candidate = date(today.year, by_month, by_day)
                        if event_date_candidate < today:
                            event_date_candidate += relativedelta(years=1)
                        event.begin = event_date_candidate
                        event.extra.append(ContentLine(name='RRULE', value=rrule_string))
                        is_recurring = True
                    except ValueError:
                        continue

            if not is_recurring:
                try:
                    event.begin = datetime.strptime(item.get('event_date', ''), "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue

            event.make_all_day()
            cal.events.add(event)
        
        if not cal.events:
            return ""
            
        ics_string = str(cal)
        return base64.b64encode(ics_string.encode('utf-8')).decode('utf-8')
