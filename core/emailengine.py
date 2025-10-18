import os
import json
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel

from config import GENERATION_MODEL, OPENAI_API_KEY, EMAIL_SENDER_ADDRESS, EMAIL_SENDER_PASSWORD, SMTP_SERVER_HOST, SMTP_SERVER_PORT
from core.schemas import EmailDraft, EmailAttachment

class EmailEngine:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def draft_email_from_result(self, result: BaseModel, context: str) -> EmailDraft:
        system_prompt = (
            "You are a senior partner at a prestigious law firm, known for clear, proactive client communication. "
            "Your task is to draft a client-facing email summarizing the results of a legal AI analysis. "
            "Your tone is authoritative, professional, and reassuring. Use HTML for formatting."
        )
        user_prompt = f"""
        Draft a client update email based on the following analysis.

        **Context of Analysis:** {context}
        **Analysis Results (JSON):**
        ```json
        {result.model_dump_json(indent=2, exclude={'email_draft', 'one_pager_pdf_base64', 'ics_file_base64', 'summary_pdf_base64'})}
        ```

        **Instructions:**
        1. Create a compelling and informative **subject line** relevant to the context of the analysis.
        2. Write the **email body** in professional HTML format. Start with a generic salutation (e.g., "Dear Valued Client,").
        3. Briefly summarize the key takeaways from the analysis in the first paragraph.
        4. Present the detailed results in a clear, readable format (e.g., using lists or tables).
        5. Conclude with a professional closing, offering further consultation.
        6. Your final output must be a single JSON object with two keys: "subject" and "body_html".
        """
        try:
            response = self.client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}
            )
            email_data = json.loads(response.choices[0].message.content)
            return EmailDraft(**email_data, delivery_status="draft_only")
        except Exception as e:
            return EmailDraft(subject="Error Generating Email Draft", body_html=f"<p>An error occurred during AI drafting: {e}</p>")

    def send_email(
        self,
        to_address: str,
        subject: str,
        body_html: str,
        attachments: Optional[List[EmailAttachment]] = None
    ) -> bool:
        if not all([EMAIL_SENDER_ADDRESS, EMAIL_SENDER_PASSWORD, SMTP_SERVER_HOST]):
            print("ERROR: SMTP email settings are not configured. Cannot send email.")
            return False
        
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_SENDER_ADDRESS
        msg['To'] = to_address
        msg['Subject'] = subject
        msg.attach(MIMEText(body_html, 'html'))
        
        if attachments:
            for att_data in attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(base64.b64decode(att_data.content_base64))
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{att_data.filename}"')
                msg.attach(part)
        
        try:
            with smtplib.SMTP(SMTP_SERVER_HOST, SMTP_SERVER_PORT) as server:
                server.starttls()
                server.login(EMAIL_SENDER_ADDRESS, EMAIL_SENDER_PASSWORD)
                server.send_message(msg)
            print(f"Email successfully sent to {to_address}")
            return True
        except Exception as e:
            print(f"Failed to send email via SMTP: {e}")
            return False