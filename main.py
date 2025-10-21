import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query, Depends, Path, Header, BackgroundTasks
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, EmailStr
import config
from core.engine import LegalQueryEngine
from core.schemas import LegalQuestionRequest, AnswerResponse, AlertsRequest, ClauseAuditResponse
from typing import List, Optional, Dict, Literal
from core.schemas import *
from core.document_processor import DocumentProcessor 
from core.analysis_engine import AnalysisEngine 
import contextlib
import json
import openai
from documents_scraper import URLS_BY_SUBJECT
from versioning import map_topics_to_subjects_llm
from scheduler import scheduler, schedule_job, run_manual_scrape
import fitz
from core.clauseaudit_engine import ClauseAuditEngine
from core.versioncompare_engine import VersionCompareEngine
from core.schemas import VersionCompareResponse, ChatSessionRecord
from core.regulatorycalender_engine import RegulatoryCalendarEngine
from core.emailengine import EmailEngine
from core.schemas import RegulatoryCalendarResponse
from core.schemas import RegulatoryAlertsResponse
from core.schemas import ChatTurnResponse, SearchRequest, SearchResponse
from core.knowledgesearch_engine import InMemorySearchEngine    
from core.intakeengine import IntakeEngine
from core.schemas import EmailAttachment, StartChatRequest
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import uuid
import traceback
import requests
from config import GENERATION_MODEL, OPENAI_API_KEY, CHAT_SESSIONS_DIR



LATEST_DIR_SYMLINK = "versions/latest"
PREVIOUS_DIR_SYMLINK = "versions/previous"
CHANGES_DIR = "regulatory_changes"
BASE_DOCUMENT_URL = "https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma"

SCHEDULER_URL = os.getenv("SCHEDULER_URL")  # e.g., https://luke-scheduler.onrender.com
SCHEDULER_SECRET = os.getenv("SCHEDULER_SECRET")
# --- FastAPI Application ---


# Create directory for storing session histories
os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    print("--- FastAPI app starting up... ---")
    schedule_job()
    scheduler.start()
    app.state.mongo_client = MongoClient(config.MONGO_DB_CONNECTION_STRING)
    app.state.db = app.state.mongo_client.LUKE
    print("--- MongoDB connection established. ---")

    print("--- Scheduler has been started for automatic weekly scrapes. ---")
    yield
    
    print("--- MongoDB connection closed. ---")
    print("--- FastAPI app shutting down... ---")
    scheduler.shutdown()
    app.state.mongo_client.close()


# Initialize the FastAPI app with the lifespan manager
app = FastAPI(  
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="API for answering Peruvian legal questions with evidence-backed citations.",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to a list of your frontend URLs for security, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers including Authorization
)
app.state.query_engine = LegalQueryEngine()
email_engine = EmailEngine() 
intake_engine = IntakeEngine()
# search_engine = InMemorySearchEngine()
@app.get("/")
def root():
    """A simple endpoint to confirm the API is running."""
    latest_path = "Not set"
    previous_path = "Not set"
    if os.path.islink(LATEST_DIR_SYMLINK):
        latest_path = os.readlink(LATEST_DIR_SYMLINK)
    if os.path.islink(PREVIOUS_DIR_SYMLINK):
        previous_path = os.readlink(PREVIOUS_DIR_SYMLINK)
        
    return {
        "message": "Scraper scheduler with automatic versioning is running.",
        "versions": {
            "latest": latest_path,
            "previous": previous_path
        }
    }


@app.post("/trigger-scraper")
def trigger_scraper():
    """Trigger the scheduler service manually."""
    if not SCHEDULER_URL or not SCHEDULER_SECRET:
        raise HTTPException(status_code=500, detail="Scheduler configuration missing.")

    try:
        res = requests.post(f"{SCHEDULER_URL}/run", json={"secret": SCHEDULER_SECRET}, timeout=10)
        return {"message": "Manual scraper triggered", "scheduler_response": res.json()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to contact scheduler: {e}")



@app.post("/answer", response_model=AnswerResponse, tags=["Flow 1 - Legal Research"])
async def get_legal_answer(
    request: LegalQuestionRequest,
    draft_email: bool = Query(False),
    send_email: bool = Query(False),
    lawyer_approved: bool = Query(True),
    recipient_email: Optional[str] = Query(None)
):
    try:
        response = app.state.query_engine.answer_question(request)
        if draft_email or send_email:
            response.email_draft = email_engine.draft_email_from_result(response, f"Legal Question: '{request.question_text}'")
            if send_email and lawyer_approved and recipient_email:
                success = email_engine.send_email(recipient_email, response.email_draft.subject, response.email_draft.body_html)
                response.email_draft.delivery_status = "sent" if success else "failed"
            elif send_email:
                response.email_draft.delivery_status = "pending_approval"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-document", response_model=DocumentSummaryResponse, tags=["Flow 2 - Document Summary"])
async def analyze_document_endpoint(
    file: UploadFile = File(...),
    topics: Optional[List[str]] = Form(None),
    draft_email: bool = Query(False),
    send_email: bool = Query(False),
    lawyer_approved: bool = Query(True),
    recipient_email: Optional[str] = Query(None)
):
    try:
        engine = AnalysisEngine(file=file, topics=topics)
        summary, findings, s_data, pdf_b64 = engine.analyze()
        response = DocumentSummaryResponse(executive_summary=summary, findings=findings, structured_data=s_data, one_pager_pdf_base64=pdf_b64)
        if draft_email or send_email:
            response.email_draft = email_engine.draft_email_from_result(response, f"Analysis of document: '{file.filename}'")
            if send_email and lawyer_approved and recipient_email:
                attachment = EmailAttachment(filename="Analysis_Summary.pdf", content_base64=pdf_b64)
                success = email_engine.send_email(recipient_email, response.email_draft.subject, response.email_draft.body_html, [attachment])
                response.email_draft.delivery_status = "sent" if success else "failed"
            elif send_email:
                response.email_draft.delivery_status = "pending_approval"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audit-clauses", response_model=ClauseAuditResponse, tags=["Flow 3 - Clause Audit"])
async def audit_clauses_endpoint(
    file: UploadFile = File(...),
    checklist: List[str] = Form(...),
    draft_email: bool = Query(False),
    send_email: bool = Query(False),
    lawyer_approved: bool = Query(True),
    recipient_email: Optional[str] = Query(None)
):
    try:
        engine = ClauseAuditEngine(file=file, checklist=checklist)
        response = engine.audit()
        # response = ClauseAuditResponse(**response_data)
        if draft_email or send_email:
            response.email_draft = email_engine.draft_email_from_result(response, f"Clause audit for document: '{file.filename}'")
            if send_email and lawyer_approved and recipient_email:
                attachment = EmailAttachment(filename="Clause_Audit_Report.pdf", content_base64=response.one_pager_pdf_base64)
                success = email_engine.send_email(recipient_email, response.email_draft.subject, response.email_draft.body_html, [attachment])
                response.email_draft.delivery_status = "sent" if success else "failed"
            elif send_email:
                response.email_draft.delivery_status = "pending_approval"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-versions", response_model=VersionCompareResponse, tags=["Flow 4 - Version Compare"])
async def compare_versions_endpoint(
    file_v1: UploadFile = File(...),
    file_v2: UploadFile = File(...),
    topic_focus: Optional[List[str]] = Form(None),
    draft_email: bool = Query(False),
    send_email: bool = Query(False),
    lawyer_approved: bool = Query(True),
    recipient_email: Optional[str] = Query(None)
):
    try:
        engine = VersionCompareEngine(file_v1=file_v1, file_v2=file_v2, topic_focus=topic_focus)
        response = engine.compare()
        # response = VersionCompareResponse(**response_data)
        if draft_email or send_email:
            response.email_draft = email_engine.draft_email_from_result(response, f"Comparison between '{file_v1.filename}' and '{file_v2.filename}'")
            if send_email and lawyer_approved and recipient_email:
                attachment = EmailAttachment(filename="Version_Comparison.pdf", content_base64=response.summary_pdf_base64)
                success = email_engine.send_email(recipient_email, response.email_draft.subject, response.email_draft.body_html, [attachment])
                response.email_draft.delivery_status = "sent" if success else "failed"
            elif send_email:
                response.email_draft.delivery_status = "pending_approval"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regulatory-alerts", response_model=RegulatoryAlertsResponse, tags=["Flow 5 - Regulatory Alerts"])
async def get_regulatory_alerts(
    request: AlertsRequest,
    draft_email: bool = Query(False),
    send_email: bool = Query(False),
    lawyer_approved: bool = Query(True),
    recipient_email: Optional[str] = Query(None)
):
    try:
        mapped_subjects = map_topics_to_subjects_llm(request.watchlist, list(URLS_BY_SUBJECT.keys()))
        if not mapped_subjects:
            raise HTTPException(status_code=404, detail="Could not map watchlist to any known subjects.")
        
        all_change_data = []
        for subject in mapped_subjects:
            subject_path = os.path.join(CHANGES_DIR, subject)
            if os.path.exists(subject_path):
                for file in os.listdir(subject_path):
                    if file.endswith('.txt'):
                        with open(os.path.join(subject_path, file), 'r', encoding='utf-8') as f:
                            doc_name = os.path.splitext(file)[0].replace('_change','').replace('_deleted','')
                            all_change_data.append({"document_name":f"{doc_name}.pdf", "summary": f.read(), "official_link": f"{BASE_DOCUMENT_URL}/{doc_name}.pdf"})
        
        if not all_change_data:
            return RegulatoryAlertsResponse(alerts=[], message="No recent regulatory changes found for the specified topics.")

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        client_profiles_str = ", ".join(request.client_profiles) if request.client_profiles else "a general business audience"
        prompt = f"Client watchlist: {request.watchlist}. Client profile: {client_profiles_str}. Detected updates: {json.dumps(all_change_data)}. Generate a JSON object with a key 'alerts', where each alert has 'summary', 'impact', 'action_items' (a list of strings), and 'official_link'."
        
        response_llm = client.chat.completions.create(model=GENERATION_MODEL, messages=[{"role":"user", "content":prompt}], response_format={"type":"json_object"})
        
        alerts_json_from_llm = json.loads(response_llm.choices[0].message.content)
        
        for alert_data in alerts_json_from_llm.get("alerts", []):
            if isinstance(alert_data.get("action_items"), str):
                print("Warning: Sanitizing 'action_items' from string to list.")
                raw_actions = alert_data["action_items"]
                alert_data["action_items"] = [item.strip() for item in raw_actions.split('\n') if item.strip()] or [raw_actions]
        
        response = RegulatoryAlertsResponse(**alerts_json_from_llm)
        
        if draft_email or send_email:
            response.email_draft = email_engine.draft_email_from_result(response, f"Regulatory alerts for topics: {', '.join(request.watchlist)}")
            if send_email and lawyer_approved and recipient_email:
                success = email_engine.send_email(recipient_email, response.email_draft.subject, response.email_draft.body_html)
                response.email_draft.delivery_status = "sent" if success else "failed"
            elif send_email:
                response.email_draft.delivery_status = "pending_approval"
        
        return response
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/extract-deadlines", response_model=RegulatoryCalendarResponse, tags=["Flow 6 - Regulatory Calendar"])
async def extract_deadlines_endpoint(
    file: UploadFile = File(...),
    draft_email: bool = Query(False),
    send_email: bool = Query(False),
    lawyer_approved: bool = Query(True),
    recipient_email: Optional[str] = Query(None)
):
    try:
        engine = RegulatoryCalendarEngine(file=file)
        
        
        response = engine.extract()
        if draft_email or send_email:
            response.email_draft = email_engine.draft_email_from_result(response, f"Key deadlines extracted from: '{file.filename}'")
            if send_email and lawyer_approved and recipient_email:
                attachment = EmailAttachment(filename="Deadlines.ics", content_base64=response.ics_file_base64)
                success = email_engine.send_email(recipient_email, response.email_draft.subject, response.email_draft.body_html, [attachment])
                response.email_draft.delivery_status = "sent" if success else "failed"
            elif send_email:
                response.email_draft.delivery_status = "pending_approval"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/start", response_model=ChatTurnResponse, tags=["Flow 8 - Conversational Intake"])
async def start_chat_session(
    user_email: EmailStr = Header(..., description="The email address of the user starting the chat."),
    goal: Literal["new_matter_intake", "policy_review"] = Form(...),
    initial_message: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    draft_email: bool = Query(False)
):
    """
    Initiates a brand new conversational session for a user.
    Always creates a new chat, even if others exist for the user.
    """
    try:
        
        response = await intake_engine.start_new_session_and_process_turn(
            user_email=user_email,
            initial_message=initial_message,
            goal=goal,
            uploaded_files=files
        )
        if response.conversation_status == "complete" and draft_email:
            response.email_draft = email_engine.draft_email_from_result(response, f"Summary of '{goal}' intake")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/turn/{session_id}", response_model=ChatTurnResponse, tags=["Flow 8 - Conversational Intake"])
async def continue_chat_turn(
    session_id: str,
    new_message: str = Form(...),
    files: List[UploadFile] = File(None),
    draft_email: bool = Query(False)
):
    """
    Continues an existing conversation turn using its unique session_id.
    """
    try:
        
        response = await intake_engine.process_turn(session_id, new_message, files)
        if response.conversation_status == "complete" and draft_email:
            session_data = intake_engine.collection.find_one({"session_id": session_id})
            goal = session_data.get("goal", "intake")
            response.email_draft = email_engine.draft_email_from_result(response, f"Summary of completed '{goal}' intake")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/history", response_model=List[ChatSessionSummary], tags=["Flow 8 - Conversational Intake"])
async def get_user_chat_history(user_email: EmailStr = Header(...)):
    """
    Retrieves a summary of all chat sessions for a specific user.
    """
    try:
        
        sessions_data = intake_engine.get_user_sessions(user_email)
        return [ChatSessionSummary(**s) for s in sessions_data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {e}")

@app.delete("/chats/{session_id}", status_code=200, tags=["Flow 8 - Conversational Intake"])
async def delete_chat_session(session_id: str, user_email: EmailStr = Header(...)):
    """
    Deletes a specific chat session for the given user to allow them to start fresh.
    """
    
    if not intake_engine.delete_user_session(session_id, user_email):
        raise HTTPException(status_code=404, detail="Session not found or user does not have permission to delete.")
    return {"message": "Chat session deleted successfully."}

@app.get("/chats/session/{session_id}", response_model=ChatSessionRecord, tags=["Flow 8 - Conversational Intake"])
async def get_single_chat_session(
    session_id: str = Path(..., description="The unique ID of the chat session to retrieve."),
    user_email: EmailStr = Header(..., description="The email address of the user who owns the chat session for verification.")
):
    """
    Retrieves the full history and details of a single chat session by its ID.
    """
    try:
        
        # This now returns either a ChatSessionRecord object or None
        session_record = intake_engine.get_session_by_id(session_id, user_email)
        
        if not session_record:
            raise HTTPException(
                status_code=404,
                detail=f"Chat session with ID '{session_id}' not found for the specified user."
            )
            
        # We are now returning a Pydantic object that already matches the response_model.
        # This is the safest and most explicit way to do it.
        return session_record
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error retrieving session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    
@app.post("/search-uploaded-documents", response_model=SearchResponse, tags=["Flow 9 - Knowledge Search (Uploaded Docs)"])
async def search_uploaded_documents_endpoint(
    # Form fields for query and files
    query: str = Form(..., description="The natural language query to search."),
    files: List[UploadFile] = File(..., description="A list of PDF or DOCX documents to search within."),
    top_k: int = Form(5, gt=0, le=20, description="The number of top results to return."),
    
    # Query parameters for universal email functionality
    draft_email: bool = Query(False, description="Set to true to generate a client email draft of the results."),
    send_email: bool = Query(False, description="Set to true to attempt sending the email."),
    lawyer_approved: bool = Query(True, description="Flag indicating if a lawyer has approved sending. Defaults to True for direct sending."),
    recipient_email: Optional[str] = Query(None, description="The recipient's email address, required if send_email is true.")
):
    """
    Performs on-the-fly semantic search across uploaded files and optionally drafts or sends an email.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    
    if send_email and not recipient_email:
        raise HTTPException(status_code=400, detail="A 'recipient_email' is required when 'send_email' is set to true.")
        
    try:
        # Initialize the engine with the uploaded files for this specific request
        engine = InMemorySearchEngine(files)
        
        # Perform the in-memory processing and indexing
        await engine.initialize()
        
        # Execute the search and get the Pydantic response object
        response = engine.search(query, top_k)
        
        # --- Full, consistent email logic ---
        if draft_email or send_email:
            file_names = ", ".join([f.filename for f in files])
            context = f"Search results for the query '{query}' within the documents: {file_names}"
            
            response.email_draft = email_engine.draft_email_from_result(response, context)
            
            if send_email and lawyer_approved and recipient_email:
                # In this flow, there are no generated PDF attachments, so we send without them.
                success = email_engine.send_email(
                    to_address=recipient_email,
                    subject=response.email_draft.subject,
                    body_html=response.email_draft.body_html
                )
                response.email_draft.delivery_status = "sent" if success else "failed"
            elif send_email:
                # If send is requested but not approved, mark it as pending
                response.email_draft.delivery_status = "pending_approval"
            
        return response

    except Exception as e:
        print(f"An unexpected error occurred during in-memory search: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")