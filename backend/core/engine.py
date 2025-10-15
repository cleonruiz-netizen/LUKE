import json
import io
import re
from typing import List, Dict, Optional
from openai import OpenAI
from supabase import create_client, Client
from .schemas import LegalQuestionRequest, AnswerResponse, Citation
from .components import HybridRetriever, PineconeVectorStore, LegalChunk
import config
import os


class LegalQueryEngine:
    def __init__(self):
        print("Initializing LegalQueryEngine...")

        # --- API Key Checks ---
        if not all([config.PINECONE_API_KEY, config.OPENAI_API_KEY]):
            raise ValueError("One or more API keys are missing. Please check your .env file.")

        # --- Supabase Setup ---
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

        if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET]):
            raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET in .env")

        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.index_dir = "indexes"  # same as ingestion.py
        self.local_index_filename = "local_index.json"

        # --- Core Components ---
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.retriever = HybridRetriever(
            openai_api_key=config.OPENAI_API_KEY,
            use_openai=config.USE_OPENAI_EMBEDDINGS,
        )
        self.vector_store = PineconeVectorStore(
            api_key=config.PINECONE_API_KEY,
            environment=config.PINECONE_ENVIRONMENT,
            index_name=config.PINECONE_INDEX_NAME,
        )

        # --- Load Index from Supabase ---
        self.chunks = self._load_index_from_supabase()

        if self.chunks:
            self.retriever.initialize_bm25(self.chunks)
            print("✅ LegalQueryEngine initialized successfully with remote index.")
        else:
            print("⚠ WARNING: No index file found in Supabase. BM25 sparse search will be disabled.")
            print("-> Please run 'python ingestion.py' to build and upload the index.")


    def _load_index_from_supabase(self) -> List[LegalChunk]:
        """Downloads and loads local_index.json directly from Supabase."""
        try:
            data = self.supabase.storage.from_(os.getenv("SUPABASE_BUCKET")).download(
                f"{self.index_dir}/{self.local_index_filename}"
            )
            index_data = json.loads(data.decode("utf-8"))
            print("☁️ Loaded local_index.json from Supabase.")
            return [LegalChunk(**chunk_data) for chunk_data in index_data.get("chunks", [])]
        except Exception as e:
            print(f"⚠ Could not load index from Supabase: {e}")
            return []


    def answer_question(self, request: LegalQuestionRequest) -> AnswerResponse:
        """Executes the full RAG pipeline based on the LUKE flow spec."""
        print(f"Received question: {request.question_text}")

        # Step 1 - Retrieve Relevant Chunks
        search_results = self.retriever.hybrid_search(
            query=request.question_text,
            vector_store=self.vector_store,
            initial_k=20,
            final_k=7,
            alpha=config.HYBRID_SEARCH_ALPHA,
        )

        if not search_results:
            return AnswerResponse(
                answer_text="I could not find any relevant legal provisions in the indexed documents to answer your question.",
                citations=[],
                official_pdf_links=[],
                confidence_flags=["no_relevant_documents_found"],
            )

        # Step 2 - Prepare Context & Citations
        context_for_llm, citations, pdf_links = self._prepare_context_and_citations(search_results)

        # Step 3 - Synthesize Final Answer
        final_answer = self._synthesize_answer_with_evidence(request.question_text, context_for_llm)

        return AnswerResponse(
            answer_text=final_answer,
            citations=citations,
            official_pdf_links=list(pdf_links),
            confidence_flags=["answer_generated_from_indexed_sources"],
        )


    def _clean_quote(self, text: str, article_number: Optional[str]) -> str:
        """Produces a lawyer-ready, concise, and properly formatted quote."""
        original_text = text

        is_current_version = '[VERSIÓN VIGENTE]' in text
        text = re.sub(r'\[VERSIÓN VIGENTE\]\s*', '', text)
        text = re.sub(r'\[Modificado por:.*?\]', '', text, flags=re.DOTALL)

        if article_number:
            pattern = re.compile(
                rf'^\s*Art[ií]culo\s+{re.escape(article_number)}[\.\-º°]*\s*[\-\.]?\s*',
                re.IGNORECASE
            )
            text = pattern.sub('', text)

        terminators = [
            'CONCORDANCIAS', 'CONCORDANCIA', 'Comuníquese', 'Dado en', 'Casa de',
            'POR TANTO', 'Artículo modificado por'
        ]
        for term in terminators:
            if term in text:
                parts = text.split(term)
                if len(parts[0].strip()) > 20:
                    text = parts[0]
                    break

        text = ' '.join(text.split()).strip()
        if len(text) > 200:
            text = text[:197] + "..."

        if is_current_version and text:
            return f'"{text}" [Versión vigente]'
        return f'"{text}"' if text else '"[Contenido no disponible]"'


    def _prepare_context_and_citations(self, search_results: List[Dict]):
        """Prepares LLM context with clear version information and clean citations."""
        context = ""
        citations = []
        pdf_links = set()

        for i, result in enumerate(search_results):
            chunk = result['chunk']

            source_name = f"{chunk.document_type or ''} N° {chunk.document_number or ''}".strip()
            if source_name == "N°":
                source_name = chunk.pdf_name.replace('_', ' ').title()

            if chunk.chunk_type == 'structural_current_version':
                source_name += " (Versión Vigente)"
            elif chunk.chunk_type == 'structural_original':
                source_name += " (Texto Original - Ver modificaciones)"

            cleaned_quote = self._clean_quote(chunk.text, chunk.article_number)

            citation = Citation(
                source_name=source_name,
                source_title=chunk.document_title or "Título no encontrado",
                article_or_section=chunk.article_number or "N/A",
                page_number=chunk.page_number,
                quote=cleaned_quote
            )

            if self.validate_citation(citation):
                citations.append(citation)

            context += f"[Evidencia {i+1}]\n"
            context += f"Fuente: {source_name}\n"
            context += f"Artículo/Sección: {chunk.article_number or 'N/A'}\n"

            if chunk.chunk_type == 'structural_current_version':
                context += "⚠️ IMPORTANTE: Esta es la versión VIGENTE del artículo.\n"
            elif chunk.chunk_type == 'structural_original':
                context += "ℹ️ NOTA: Este es el texto original. Puede haber una versión modificada.\n"

            context += f"Contenido: {cleaned_quote}\n\n"

            pdf_links.add(
                f"https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/{chunk.pdf_name.replace('_', '-')}"
            )

        return context, citations, list(pdf_links)

    def validate_citation(self, citation: Citation) -> bool:
        """Validates that a citation has meaningful content."""
        if not citation.quote or citation.quote in ['"..."', '""', '"[Contenido no disponible]"']:
            return False
        if len(citation.quote.strip('"')) < 10:
            return False
        return True

    def deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Removes duplicate citations based on article number and source."""
        seen = set()
        unique_citations = []
        for citation in citations:
            key = (citation.source_name, citation.article_or_section)
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        return unique_citations


    def _synthesize_answer_with_evidence(self, question: str, context: str) -> str:
        """Uses a powerful LLM to generate an answer based ONLY on the provided evidence."""
        system_prompt = """
You are LUKE, an expert AI legal assistant for Peruvian lawyers.

CRITICAL RULES FOR CITATIONS:
1. **Prioritize Current Law**: If both "Texto Original" and "Versión Vigente" exist, always use "Versión Vigente".
2. **Be Explicit About Modifications**: If citing modified law, mention the modification.
3. **Evidence-Only**: Never invent legal provisions. State clearly if evidence is insufficient.
4. **Synthesis Required**: Don't list evidence; weave it into a coherent legal analysis.
5. **No Internal References**: Never mention "Evidencia 1" or "according to the document".
6. **Professional Tone**: Write as a senior legal researcher would.
"""

        user_prompt = f"""
Pregunta del usuario: "{question}"

Evidencia recuperada:
---
{context}
---

Instrucciones:
- Sintetiza la evidencia en una respuesta coherente y completa.
- Si hay versiones originales y modificadas del mismo artículo, usa SOLO la versión vigente.
- Si la evidencia es insuficiente, indícalo claramente.
"""
        try:
            response = self.openai_client.chat.completions.create(
                model=config.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return "An error occurred while generating the final answer from the retrieved evidence."
