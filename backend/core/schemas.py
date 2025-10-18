from typing import Dict, List, Optional, Literal, Any, Annotated
from datetime import datetime
from pydantic import BaseModel, Field, BeforeValidator, EmailStr
from bson import ObjectId

class EmailDraft(BaseModel):
    subject: str
    body_html: str
    delivery_status: Literal["draft_only", "not_requested", "sent", "failed", "pending_approval"] = "not_requested"

class EmailAttachment(BaseModel):
    filename: str
    content_base64: str

class Citation(BaseModel):
    source_name: str
    source_title: str
    article_or_section: str
    page_number: int
    quote: str

class AnswerResponse(BaseModel):
    answer_text: str
    citations: List[Citation]
    official_pdf_links: List[str]
    confidence_flags: List[str] = []
    email_draft: Optional[EmailDraft] = None

class StructuredData(BaseModel):
    parties: Optional[List[str]] = None
    effective_date: Optional[str] = None
    termination_date: Optional[str] = None
    key_deadlines: Optional[Dict[str, str]] = None
    monetary_amounts: Optional[List[str]] = None

class Finding(BaseModel):
    topic: str
    summary: str
    page_number: int
    quote: str

class DocumentSummaryResponse(BaseModel):
    executive_summary: str
    findings: List[Finding]
    structured_data: StructuredData
    one_pager_pdf_base64: str
    quality_flags: List[str] = []
    email_draft: Optional[EmailDraft] = None

class ClauseAuditItem(BaseModel):
    clause_name: str
    status: Literal["PRESENT", "DEVIATES", "MISSING"]
    summary: str
    page_number: Optional[int]
    quote: str

class ClauseAuditResponse(BaseModel):
    audit_table: List[ClauseAuditItem]
    one_pager_pdf_base64: str
    quality_flags: List[str] = []
    email_draft: Optional[EmailDraft] = None

class ChangeLogEntry(BaseModel):
    change_type: str
    section_title: str
    old_text: str
    new_text: str
    summary: str
    impact_explanation: str
    impact_level: str
    category: str
    negotiation_points: List[str] = []
    page_v1: Optional[int]
    page_v2: Optional[int]

class VersionCompareResponse(BaseModel):
    change_log: List[ChangeLogEntry]
    summary_pdf_base64: str
    quality_flags: List[str] = []
    total_changes: int
    email_draft: Optional[EmailDraft] = None

class CalendarItem(BaseModel):
    title: str
    is_plannable: bool
    event_date: str
    category: str
    responsible_party: Optional[str] = None
    citation: str
    recurrence: Optional[dict] = None

class RegulatoryCalendarResponse(BaseModel):
    calendar_items: List[CalendarItem]
    ics_file_base64: str
    quality_flags: List[str] = []
    email_draft: Optional[EmailDraft] = None

class Alert(BaseModel):
    summary: str
    impact: str
    action_items: List[str]
    official_link: str

class RegulatoryAlertsResponse(BaseModel):
    alerts: List[Alert]
    message: Optional[str] = None
    email_draft: Optional[EmailDraft] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class StartChatRequest(BaseModel):
    initial_message: str
    goal: Literal["new_matter_intake", "policy_review"]

class ChatTurnResponse(BaseModel):
    session_id: str
    ai_response: ChatMessage
    conversation_status: Literal["in_progress", "awaiting_document", "complete"]
    final_work_product: Optional[Dict[str, Any]] = None
    updated_history: List[ChatMessage]
    email_draft: Optional[EmailDraft] = None

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v,field=None):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema=None, handler=None):
        return {"type": "string"}

class ChatSessionRecord(BaseModel):
    id: PyObjectId = Field(..., alias="_id")
    session_id: str
    user_email: EmailStr
    goal: str
    history: List[ChatMessage]
    created_at: datetime
    last_updated: datetime

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }

class ChatSessionSummary(BaseModel):
    session_id: str
    user_email: EmailStr
    goal: str
    created_at: datetime
    last_updated: datetime

class SearchResultItem(BaseModel):
    file: str
    page: int
    snippet: str
    link_to_page: str
    confidence: float

class SearchResponse(BaseModel):
    ranked_results: List[SearchResultItem]
    email_draft: Optional[EmailDraft] = None

class LegalQuestionRequest(BaseModel):
    question_text: str = Field(..., min_length=10)
    context_tags: Optional[List[str]] = None

class AlertsRequest(BaseModel):
    watchlist: list[str]
    client_profiles: list[str] = []

class ChatIntakeRequest(BaseModel):
    history: List[ChatMessage]
    new_message: str
    goal: Literal["new_matter_intake", "policy_review"] = "new_matter_intake"

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(5, gt=0, le=20)

class EmailAttachment(BaseModel):
    filename: str
    content_base64: str


# from pydantic import BaseModel, Field
# from typing import Dict, List, Optional, Literal, Any

# class EmailDraft(BaseModel):
#     subject: str
#     body_html: str
#     delivery_status: Literal["draft_only", "not_requested"] = "not_requested"

# # --- API Request Models ---
# class LegalQuestionRequest(BaseModel):
#     question_text: str = Field(..., min_length=10, description="The natural-language legal question.")
#     context_tags: Optional[List[str]] = Field(None, description="Optional tags for filtering (e.g., 'tax', 'labor').")
#     uploaded_files: Optional[List[str]] = Field(None, description="Placeholder for future file upload feature.")

# # --- API Response Models ---
# class Citation(BaseModel):
#     source_name: str = Field(..., description="The official name of the source law (e.g., 'Ley N° 25211').")
#     source_title: str = Field(..., description="The descriptive title of the legal document.")
#     article_or_section: str = Field(..., description="The specific article or section number being cited.")
#     page_number: int = Field(..., description="The page number in the PDF where the evidence was found.")
#     quote: str = Field(..., description="A clean, concise quote of the article's content that supports the answer.")

# class AnswerResponse(BaseModel):
#     answer_text: str = Field(..., description="The synthesized, lawyer-grade answer based on evidence.")
#     citations: List[Citation] = Field(..., description="A list of all evidence-based citations used for the answer.")
#     official_pdf_links: List[str] = Field(..., description="List of official URLs to the cited legal documents.")
#     confidence_flags: List[str] = Field([], description="Flags for potential issues, like ambiguities or missing information.")
#     email_draft: Optional[EmailDraft] = None

# class StructuredData(BaseModel):
#     parties: Optional[List[str]] = Field(None, description="Names of the primary parties involved (e.g., companies, individuals).")
#     effective_date: Optional[str] = Field(None, description="The stated effective date of the document.")
#     termination_date: Optional[str] = Field(None, description="The termination or expiration date, if specified.")
#     key_deadlines: Optional[Dict[str, str]] = Field(None, description="A dictionary of important deadlines and their dates.")
#     monetary_amounts: Optional[List[str]] = Field(None, description="Significant monetary values mentioned (e.g., contract value, penalties).")

# class Finding(BaseModel):
#     topic: str = Field(..., description="The topic of the finding (e.g., 'Confidentiality', 'Termination Clause').")
#     summary: str = Field(..., description="A concise summary of the key clause or risk identified.")
#     page_number: int = Field(..., description="The page number where the evidence for the finding is located.")
#     quote: str = Field(..., description="The exact quote from the document that supports the finding.")

# class DocumentSummaryResponse(BaseModel):
#     executive_summary: str = Field(..., description="A one-page executive summary of the document, written in an expert tone.")
#     findings: List[Finding] = Field(..., description="A list of detailed findings, especially on requested topics.")
#     structured_data: StructuredData = Field(..., description="Key data points extracted from the document in a structured format.")
#     one_pager_pdf_base64: str = Field(..., description="The generated one-page summary report, encoded as a Base64 string.")
#     quality_flags: List[str] = Field([], description="A list of warnings, such as 'low_ocr_confidence' if the document appears to be a poor quality scan.")
#     email_draft: Optional[EmailDraft] = None

# class AlertsRequest(BaseModel):
#     watchlist: list[str]  # Renamed from client_topics for clarity
#     client_profiles: list[str] = []

# class SearchRequest(BaseModel):
#     query: str = Field(..., min_length=3, description="The natural language query to search the knowledge base.")
#     top_k: int = Field(5, gt=0, le=20, description="The number of top results to return.")

# class SearchResultItem(BaseModel):
#     file: str = Field(..., description="The name of the source document.")
#     page: int = Field(..., description="The page number within the document where the result was found.")
#     snippet: str = Field(..., description="The relevant text passage that matched the query.")
#     link_to_page: str = Field(..., description="A link to open the document directly to the relevant page.")
#     confidence: float = Field(..., description="The relevance score of the result (typically 0.0 to 1.0).")

# class SearchResponse(BaseModel):
#     ranked_results: List[SearchResultItem]

# class ClauseAuditItem(BaseModel):
#     clause_name: str
#     status: Literal["PRESENT", "DEVIATES", "MISSING"]
#     summary: str
#     page_number: Optional[int]
#     quote: str

# class ClauseAuditResponse(BaseModel):
#     audit_table: List[ClauseAuditItem]
#     one_pager_pdf_base64: str
#     quality_flags: List[str] = []
#     email_draft: Optional[EmailDraft] = None


# class ChangeLogEntry(BaseModel):
#     change_type: str  # "addition" | "deletion" | "modification"
#     section_title: str
#     old_text: str
#     new_text: str
#     summary: str
#     impact_explanation: str
#     impact_level: str  # "critical" | "high" | "medium" | "low"
#     category: str  # "economics" | "risk" | "timing" | "scope" | "governance" | "compliance" | "other"
#     negotiation_points: List[str] = []
#     page_v1: Optional[int]
#     page_v2: Optional[int]


# class VersionCompareResponse(BaseModel):
#     change_log: List[ChangeLogEntry]
#     summary_pdf_base64: str  # Base64-encoded PDF
#     quality_flags: List[str] = []
#     total_changes: int
#     email_draft: Optional[EmailDraft] = None


# class CalendarItem(BaseModel):
#     title: str = Field(..., description="A clear, human-readable summary of the event or deadline.")
#     is_plannable: bool = Field(..., description="True if the deadline can be placed on a calendar.")
#     event_date: str = Field(..., description="The specific date (YYYY-MM-DD) or a recurring/relative rule.")
#     category: str = Field(..., description="The type of event, e.g., 'Filing', 'Notice', 'Renewal'.")
#     responsible_party: Optional[str] = Field(None, description="The party responsible for the action.")
#     citation: str = Field(..., description="The exact quote from the document that establishes the deadline.")
#     recurrence: Optional[dict] = Field(None, description="Structured recurrence data, if applicable.")

# class RegulatoryCalendarResponse(BaseModel):
#     calendar_items: List[CalendarItem] = Field(..., description="A list of all extracted deadlines.")
#     ics_file_base64: str = Field(..., description="A Base64 encoded .ics file for plannable events.")
#     quality_flags: List[str] = Field([], description="Warnings, such as 'low_ocr_confidence'.")
#     email_draft: Optional[EmailDraft] = None

# class Alert(BaseModel):
#     summary: str
#     impact: str
#     action_items: List[str]
#     official_link: str


# class RegulatoryAlertsResponse(BaseModel):
#     alerts: List[Alert]
#     email_draft: Optional[EmailDraft] = None # Make this optional
#     message: Optional[str] = None
#     email_draft: Optional[EmailDraft] = None

# class ChatMessage(BaseModel):
#     role: Literal["user", "assistant", "system"]
#     content: str

# class ChatIntakeRequest(BaseModel):
#     history: List[ChatMessage]
#     new_message: str
#     goal: Literal["new_matter_intake", "policy_review"] = "new_matter_intake"

# class ChatIntakeResponse(BaseModel):
#     ai_response: ChatMessage
#     conversation_status: Literal["in_progress", "awaiting_document", "complete"]
#     final_work_product: Optional[Dict[str, Any]] = None
#     updated_history: List[ChatMessage]
#     email_draft: Optional[EmailDraft] = None
