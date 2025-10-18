import os
import json
from typing import List, Optional

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec
from core.schemas import SearchRequest, SearchResultItem, SearchResponse
from fastapi import FastAPI, HTTPException, Body, UploadFile
from core.components import EnhancedLegalChunker, HybridRetriever, LegalChunk

import numpy as np
import fitz

from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT,PINECONE_INDEX_NAME, LOCAL_INDEX_PATH, EMBEDDING_MODEL_OPENAI
class InMemorySearchEngine:
    """
    Performs on-the-fly semantic search across a list of user-uploaded documents.
    """
    def __init__(self, files: List[UploadFile]):
        self.files = files
        self.chunker = EnhancedLegalChunker()
        self.retriever = HybridRetriever(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            use_openai=True
        )
        self.all_chunks: List[LegalChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

    async def _process_and_chunk_files(self):
        """Processes all uploaded files, chunks them, and stores them in memory."""
        print(f"Processing and chunking {len(self.files)} uploaded documents...")
        for file in self.files:
            await file.seek(0)
            file_content = await file.read()
            try:
                with fitz.open(stream=file_content, filetype="pdf") as doc:
                    full_text = "".join(page.get_text() for page in doc)
                    doc_metadata = self.chunker.extract_document_metadata(full_text, file.filename)
                    for page_num, page in enumerate(doc, 1):
                        page_text = page.get_text()
                        if page_text.strip():
                            page_chunks = self.chunker.create_contextual_chunks(
                                page_text, file.filename, file.filename, page_num, doc_metadata
                            )
                            self.all_chunks.extend(page_chunks)
            except Exception as e:
                print(f"Warning: Failed to process file {file.filename}. Error: {e}")
        print(f"Generated a total of {len(self.all_chunks)} chunks.")

    def _build_in_memory_indexes(self):
        """Builds vector and BM25 indexes in memory from the processed chunks."""
        if not self.all_chunks:
            return

        print("Building in-memory BM25 index...")
        self.retriever.initialize_bm25(self.all_chunks)

        print("Generating embeddings for all chunks...")
        chunk_texts = [chunk.text for chunk in self.all_chunks]
        self.chunk_embeddings = self.retriever.batch_embed(chunk_texts)
        print("Embeddings generated successfully.")

    async def initialize(self):
        """Asynchronous initializer to perform all setup tasks."""
        await self._process_and_chunk_files()
        self._build_in_memory_indexes()

    def _in_memory_vector_search(self, query_vector: np.ndarray, top_k: int) -> List[dict]:
        """Performs a simple cosine similarity search against the in-memory embeddings."""
        if self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
            return []
        
        # Normalize vectors for cosine similarity
        query_vector_norm = query_vector / np.linalg.norm(query_vector)
        chunk_embeddings_norm = self.chunk_embeddings / np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.dot(chunk_embeddings_norm, query_vector_norm)
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {"id": self.all_chunks[i].chunk_id, "score": float(similarities[i])}
            for i in top_indices
        ]

    def search(self, query: str, top_k: int) -> SearchResponse:
        """Executes the full in-memory hybrid search pipeline."""
        if not self.all_chunks:
            return SearchResponse(ranked_results=[])

        # Dense search against in-memory vectors
        query_embedding = self.retriever.get_embedding(query)
        dense_results = self._in_memory_vector_search(query_embedding, top_k=top_k * 3)

        # Sparse (BM25) search
        sparse_results = self.retriever.bm25_search(query, top_k=top_k * 3)

        # Combine scores (Reciprocal Rank Fusion or simple weighted sum)
        combined_scores = {}
        for res in dense_results:
            combined_scores[res['id']] = res['score'] * 0.6 # Dense weight
        
        for idx, score in sparse_results:
            chunk_id = self.all_chunks[idx].chunk_id
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += (score / 10.0) * 0.4 # Sparse weight (normalized heuristically)
            else:
                combined_scores[chunk_id] = (score / 10.0) * 0.4

        # Get top candidate IDs and hydrate
        top_candidate_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        id_to_chunk = {c.chunk_id: c for c in self.all_chunks}
        
        final_results = []
        for chunk_id in top_candidate_ids:
            chunk = id_to_chunk.get(chunk_id)
            if chunk:
                final_results.append(SearchResultItem(
                    file=chunk.pdf_name,
                    page=chunk.page_number,
                    snippet=chunk.text,
                    link_to_page=f"{chunk.pdf_name}#page={chunk.page_number}",
                    confidence=combined_scores.get(chunk_id, 0.0)
                ))
        
        return SearchResponse(ranked_results=final_results)