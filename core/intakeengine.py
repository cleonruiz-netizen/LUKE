import os
import json
import uuid
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone

import fitz
from openai import OpenAI
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ConfigurationError
from fastapi import UploadFile, HTTPException

from core.schemas import ChatMessage, ChatTurnResponse, ChatSessionRecord
from core.regulatorycalender_engine import RegulatoryCalendarEngine
import config

class IntakeEngine:
    
    def _is_legal_research_query(self, message: str, uploaded_files: Optional[List[UploadFile]]) -> bool:
        """
        Uses OpenAI to classify if the user's message is a legal research question, based only on the content of the query.
        Returns True if AI thinks it's a legal research query, else False.
        """
        system_prompt = (
            "You are an expert legal assistant. "
            "Analyze the following user message and answer ONLY 'yes' or 'no': "
            "Is this message a legal research question (e.g., asking for legal information, analysis, summary, or advice about laws, regulations, clauses, or documents)? "
            "Do NOT consider whether files are uploaded. Only analyze the content of the message."
        )
        user_prompt = f"Message: {message}"
        response = self.client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        raw_answer = response.choices[0].message.content.strip()
        print(f"[AI Legal Research Query Detection] Raw response: {raw_answer}")
        answer = raw_answer.lower()
        return answer.startswith("yes")
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        if not config.MONGO_DB_CONNECTION_STRING:
            raise RuntimeError("MONGO_DB_CONNECTION_STRING is not set.")
        self.db_client = MongoClient(config.MONGO_DB_CONNECTION_STRING)
        self.db = self.db_client.luke_api
        self.collection = self.db.chat_sessions
        self.collection.create_index("user_email")
        self.collection.create_index("session_id", unique=True)
        print("--- MongoDB connection established and indexes ensured. ---")

    def get_user_sessions(self, user_email: str) -> List[Dict]:
        sessions = self.collection.find(
            {"user_email": user_email},
            {"history": 0}
        ).sort("last_updated", -1)
        return list(sessions)

    def delete_user_session(self, session_id: str, user_email: str) -> bool:
        result = self.collection.delete_one({"session_id": session_id, "user_email": user_email})
        deleted = result.deleted_count > 0
        print(f"Session {session_id} for user {user_email} deleted. Success: {deleted}")
        return deleted

    def _load_history(self, session_id: str) -> List[ChatMessage]:
        session_data = self.collection.find_one({"session_id": session_id})
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Chat session with ID '{session_id}' not found.")
        return [ChatMessage(**msg) for msg in session_data.get("history", [])]

    def _save_history(self, session_id: str, history: List[ChatMessage]):
        self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"history": [msg.model_dump() for msg in history], "last_updated": datetime.now(timezone.utc)}}
        )

    def _generate_final_product(self, history: List[ChatMessage], goal: str) -> Dict[str, Any]:
        full_transcript = "\n".join([f"{msg.role.title()}: {msg.content}" for msg in history])
        system_prompt = "You are a senior lawyer synthesizing a conversation into a final work product. Determine the correct output format (memo, checklist, calendar) and generate it."
        user_prompt = f"""Based on the following intake conversation for a '{goal}', generate the most appropriate final work product. If calendar events are needed, structure them with `is_plannable`, `event_date`, `recurrence`, etc., just like our calendar engine expects. CONVERSATION TRANSCRIPT: {full_transcript} Your response must be a valid JSON object."""
        
        response = self.client.chat.completions.create(model=config.GENERATION_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format={"type": "json_object"})
        final_product = json.loads(response.choices[0].message.content)

        calendar_items_data = final_product.get("calendar_items")
        if isinstance(calendar_items_data, list) and calendar_items_data:
            try:
                ics_base64 = RegulatoryCalendarEngine._create_ics_file(calendar_items_data)
                if ics_base64:
                    final_product["ics_file_base64"] = ics_base64
            except Exception as e:
                final_product["ics_generation_error"] = str(e)

        return final_product

    async def start_new_session_and_process_turn(
        self,
        user_email: str,
        initial_message: str,
        goal: str,
        uploaded_files: Optional[List[UploadFile]] = None
    ) -> ChatTurnResponse:
        new_session_id = str(uuid.uuid4())
        self.collection.insert_one({
            "session_id": new_session_id,
            "user_email": user_email,
            "goal": goal,
            "history": [],
            "created_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc)
        })
        return await self.process_turn(new_session_id, initial_message, uploaded_files)

    async def process_turn(
        self,
        session_id: str,
        new_message: str,
        uploaded_files: Optional[List[UploadFile]] = None
    ) -> ChatTurnResponse:
        history = self._load_history(session_id)
        session_data = self.collection.find_one({"session_id": session_id})
        goal = session_data.get("goal", "new_matter_intake")

        # Check if this is a legal research query about uploaded docs
        if self._is_legal_research_query(new_message, uploaded_files):
            # Use the first file for demo; you can extend to multiple files
            file = uploaded_files[0] if uploaded_files else None
            from core.schemas import LegalQuestionRequest
            from main import app  # Import FastAPI app to access query_engine
            request = LegalQuestionRequest(question_text=new_message, file=file)
            response = app.state.query_engine.answer_question(request)

            # Compose a rich, natural chat response including all relevant info
            chat_content = f"""
{response.answer_text}

"""
            if response.citations:
                chat_content += "\nCitations:\n"
                for c in response.citations:
                    chat_content += f"- {c.source_name} ({c.source_title}), Section: {c.article_or_section}, Page: {c.page_number}\n  Quote: {c.quote}\n"
            if response.official_pdf_links:
                chat_content += "\nOfficial PDF Links:\n"
                for link in response.official_pdf_links:
                    chat_content += f"- {link}\n"
            if response.confidence_flags:
                chat_content += "\nConfidence Flags:\n"
                for flag in response.confidence_flags:
                    chat_content += f"- {flag}\n"

            # Add a friendly summary for chat
            chat_content += "\nIf you need further details or clarification, feel free to ask!"

            ai_response_message = ChatMessage(role="assistant", content=chat_content.strip())
            final_history = history + [ChatMessage(role="user", content=new_message), ai_response_message]
            self._save_history(session_id, final_history)
            return ChatTurnResponse(
                session_id=session_id,
                ai_response=ai_response_message,
                conversation_status="complete" if response else "in_progress",
                final_work_product=None,
                updated_history=final_history
            )

        # Otherwise, normal chatbot flow
        user_content = new_message
        if uploaded_files:
            for file in uploaded_files:
                await file.seek(0)
                file_content = await file.read()
                try:
                    with fitz.open(stream=file_content, filetype="pdf") as doc:
                        file_text = "".join(page.get_text() for page in doc)
                    user_content += f"\n\n--- CONTENT FROM UPLOADED FILE '{file.filename}' ---\n{file_text[:5000]}..."
                except Exception as e:
                    user_content += f"\n\n[Could not process uploaded file '{file.filename}'. Error: {e}]"

        updated_history = history + [ChatMessage(role="user", content=user_content)]

        system_prompt = f"""You are LUKE, an expert paralegal conducting a '{goal}'. Guide the user with one question at a time. If you need a document, end with `[AWAITING_DOCUMENT]`. When you have all facts, end with `[READY_TO_FINALIZE]`."""

        response = self.client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[ChatMessage(role="system", content=system_prompt).model_dump()] + [msg.model_dump() for msg in updated_history]
        )

        ai_content = response.choices[0].message.content
        status = "in_progress"
        final_product = None

        if "[AWAITING_DOCUMENT]" in ai_content:
            status = "awaiting_document"
            ai_content = ai_content.replace("[AWAITING_DOCUMENT]", "").strip()
        elif "[READY_TO_FINALIZE]" in ai_content:
            status = "complete"
            ai_content = ai_content.replace("[READY_TO_FINALIZE]", "").strip()
            final_product = self._generate_final_product(updated_history, goal)

        ai_response_message = ChatMessage(role="assistant", content=ai_content)
        final_history = updated_history + [ai_response_message]

        self._save_history(session_id, final_history)

        return ChatTurnResponse(
            session_id=session_id,
            ai_response=ai_response_message,
            conversation_status=status,
            final_work_product=final_product,
            updated_history=final_history
        )
    
    def get_session_by_id(self, session_id: str, user_email: str) -> Optional[ChatSessionRecord]:
        """
        Retrieves the full details of a single chat session as a Pydantic object,
        ensuring it belongs to the specified user.

        Args:
            session_id (str): The unique ID of the session to retrieve.
            user_email (str): The email of the user who owns the session.

        Returns:
            Optional[ChatSessionRecord]: The validated Pydantic model of the session, or None if not found.
        """
        print(f"Attempting to retrieve session '{session_id}' for user '{user_email}'...")
        session_data = self.collection.find_one({
            "session_id": session_id,
            "user_email": user_email
        })

        if session_data:
            # If data is found, create and return the Pydantic model instance.
            # This validates the data retrieved from the database.
            return ChatSessionRecord(**session_data)
        
        # If no data is found, return None.
        return None