# core/components.py

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import cohere
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


@dataclass
class LegalChunk:
    """Enhanced legal chunk with comprehensive metadata"""
    chunk_id: str
    text: str
    pdf_name: str
    pdf_path: str
    page_number: int
    chunk_type: str  # 'semantic', 'structural', 'hybrid'
    
    # Legal structure
    document_title: Optional[str] = None
    document_type: Optional[str] = None
    document_number: Optional[str] = None
    promulgation_date: Optional[str] = None
    
    # Hierarchical position
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    article_number: Optional[str] = None
    section_number: Optional[str] = None
    
    # References and context
    legal_references: List[str] = None
    cross_references: List[str] = None
    parent_context: Optional[str] = None
    
    # Text characteristics
    char_count: int = 0
    word_count: int = 0
    
    # Embedding metadata
    embedding_model: Optional[str] = None
    
    def __post_init__(self):
        if self.legal_references is None:
            self.legal_references = []
        if self.cross_references is None:
            self.cross_references = []
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())


class EnhancedLegalChunker:
    """Advanced legal document chunker with structure awareness"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Legal patterns for Peruvian documents
        self.patterns = {
            'document_type': r'(DECRETO\s+(?:LEGISLATIVO|SUPREMO)|LEY|REGLAMENTO|RESOLUCI[OÓ]N\s+(?:MINISTERIAL|DIRECTORAL))\s+N[º°]\s*(\d+[-\w]*)',
            'chapter': r'CAP[IÍ]TULO\s+([IVXLCDM]+)\s*[\n\r]+(.*?)(?=\n|$)',
            'date': r'(?:(?:Lima|Casa\s+de\s+Gobierno).*?)?(\d{1,2}).*?(?:de\s+)?(\w+).*?(?:de\s+)?(\d{4})',
            'law_ref': r'(?:Ley|Decreto\s+(?:Legislativo|Supremo)|D\.?[LS]\.?)\s+N[º°]\s*(\d+[\-\w]*)',
            'article_ref': r'art[ií]culo\s+(\d+)',
            'concordance': r'CONCORDANCIA[S]?\s*:\s*(.*?)(?=\n\n|\Z)',
        }
    
    def extract_document_metadata(self, text: str, pdf_name: str) -> Dict[str, Any]:
        """Extract document-level metadata"""
        metadata = {}
        doc_match = re.search(self.patterns['document_type'], text, re.IGNORECASE)
        if doc_match:
            metadata['document_type'] = doc_match.group(1).strip()
            metadata['document_number'] = doc_match.group(2).strip()
        
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 20 and not re.match(r'^[\d\s\-]+$', line):
                metadata['document_title'] = line[:200]
                break
        
        date_match = re.search(self.patterns['date'], text, re.IGNORECASE)
        if date_match:
            try:
                day, month, year = date_match.groups()
                metadata['promulgation_date'] = f"{day} de {month} de {year}"
            except: pass
        
        return metadata

    def extract_hierarchical_structure(self, text: str) -> List[Dict[str, Any]]:
        """Extract document structure (chapters, articles, sections)"""
        structure = []
        
        for chapter_match in re.finditer(self.patterns['chapter'], text, re.IGNORECASE | re.DOTALL):
            structure.append({
                'type': 'chapter', 'number': chapter_match.group(1),
                'title': chapter_match.group(2).strip(), 'start_pos': chapter_match.start(),
                'end_pos': chapter_match.end()
            })
        
        # --- SOLUTION IMPLEMENTED HERE ---
        # This regex is more precise. It uses '^' to anchor the search to the
        # beginning of a line, ensuring we only capture defining article titles.
        # The (?=\n^Art[ií]culo|\Z) part is a lookahead that ensures the chunk
        # ends right before the next article or at the end of the text.
        self.patterns['article'] = r'^Art[ií]culo\s+(\d+)[\.\-º°]*\s*[\-\.]?\s*(.*?)(?=\n^Art[ií]culo|\Z)'

        # The re.MULTILINE flag is crucial for the '^' anchor to work on each line.
        for article_match in re.finditer(self.patterns['article'], text, re.IGNORECASE | re.DOTALL | re.MULTILINE):
            structure.append({
                'type': 'article', 'number': article_match.group(1),
                'content': article_match.group(2).strip(), 'start_pos': article_match.start(),
                'end_pos': article_match.end()
            })
        
        structure.sort(key=lambda x: x['start_pos'])
        return structure
    
    def extract_legal_references(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract legal references and cross-references"""
        legal_refs = [m.group(0) for m in re.finditer(self.patterns['law_ref'], text, re.IGNORECASE)]
        cross_refs = [f"Artículo {m.group(1)}" for m in re.finditer(self.patterns['article_ref'], text, re.IGNORECASE)]
        return list(set(legal_refs)), list(set(cross_refs))
    # def extract_hierarchical_structure(self, text: str) -> List[Dict[str, Any]]:
    #     """Extract document structure with better boundary detection."""
    #     structure = []

    #     # Chapters (unchanged)
    #     for chapter_match in re.finditer(self.patterns['chapter'], text, re.IGNORECASE | re.DOTALL):
    #         structure.append({
    #             'type': 'chapter',
    #             'number': chapter_match.group(1),
    #             'title': chapter_match.group(2).strip(),
    #             'start_pos': chapter_match.start(),
    #             'end_pos': chapter_match.end()
    #         })

    #     # IMPROVED: Article extraction that stops at known terminators
    #     # This regex captures the article number and content, but stops at:
    #     # - Next article
    #     # - Modification notes (*)
    #     # - CONCORDANCIAS section
    #     # - End of text
    #     article_pattern = r'''
    #         ^Art[ií]culo\s+(\d+)[\.\-º°]*\s*[\-\.]?\s*  # Article header
    #         (.*?)                                          # Content (non-greedy)
    #         (?=                                            # Stop before:
    #             \n^Art[ií]culo\s+\d+                      # Next article
    #             |\n\(\*\)\s*Art[ií]culo                   # Modification note
    #             |\nCONCORDANCIAS?\s*:                      # Concordances section
    #             |\Z                                        # End of text
    #         )
    #     '''

    #     for article_match in re.finditer(article_pattern, text,
    #                                      re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE):
    #         article_content = article_match.group(2).strip()

    #         # Additional cleanup: remove trailing signatures/dates
    #         article_content = re.sub(
    #             r'\n(?:Comuníquese|Dado en|Casa de|POR TANTO).*$', 
    #             '', 
    #             article_content, 
    #             flags=re.DOTALL
    #         )

    #         structure.append({
    #             'type': 'article',
    #             'number': article_match.group(1),
    #             'content': article_content.strip(),
    #             'start_pos': article_match.start(),
    #             'end_pos': article_match.end()
    #         })

    #     structure.sort(key=lambda x: x['start_pos'])
    #     return structure

    


    # In core/components.py
# (This method goes inside the EnhancedLegalChunker class)

    def create_contextual_chunks(self, text: str, pdf_name: str, pdf_path: str, 
                                 page_num: int, doc_metadata: Dict) -> List[LegalChunk]:
        """
        Creates contextual chunks, intelligently separating original articles from
        explicit modification notes found in the text.
        """
        chunks = []
        
        # --- Stage 1: Isolate and Separate Modification Notes ---
        # This pattern finds the entire block of a modification note.
        mod_pattern = re.compile(r'(\(\*\)\s*Artículo modificado por.*?)(?=\n^Art[ií]culo|\Z)', re.IGNORECASE | re.DOTALL | re.MULTILINE)
        
        main_text = text
        modification_blocks = []
        for match in mod_pattern.finditer(text):
            block_text = match.group(1).strip()
            modification_blocks.append(block_text)
            # Remove the modification block from the main text to prevent reprocessing.
            main_text = main_text.replace(block_text, '')

        # --- Stage 2: Process the Main Body for Standard Articles ---
        structure = self.extract_hierarchical_structure(main_text)
        current_chapter_info = {'number': None, 'title': None}

        for element in structure:
            if element['type'] == 'chapter':
                current_chapter_info = {'number': element['number'], 'title': element['title']}
                continue
            
            if element['type'] == 'article':
                article_num = element['number']
                # The content is already cleaned by the new regex in extract_hierarchical_structure
                article_content = f"Artículo {article_num}.- {element['content']}"
                
                legal_refs, cross_refs = self.extract_legal_references(article_content)
                
                chunk = LegalChunk(
                    chunk_id=self._generate_chunk_id(pdf_name, page_num, article_num, 0),
                    text=article_content,
                    pdf_name=pdf_name, pdf_path=pdf_path, page_number=page_num,
                    chunk_type='structural',
                    document_title=doc_metadata.get('document_title'),
                    document_type=doc_metadata.get('document_type'),
                    document_number=doc_metadata.get('document_number'),
                    promulgation_date=doc_metadata.get('promulgation_date'),
                    chapter_number=current_chapter_info['number'],
                    chapter_title=current_chapter_info['title'],
                    article_number=article_num,
                    legal_references=legal_refs, cross_references=cross_refs,
                    parent_context=current_chapter_info['title']
                )
                chunks.append(chunk)

        # --- Stage 3: Process Each Isolated Modification Block ---
        for i, block in enumerate(modification_blocks):
            # This pattern is specifically designed to get the clean text of the modified article.
            text_match = re.search(r'cuyo texto es el siguiente:\s*["“](.*?)["”]', block, re.IGNORECASE | re.DOTALL)
            
            if text_match:
                modified_text = text_match.group(1).strip()
                # Heuristic: Assume it modifies the last found article number or default to '1'.
                # A more advanced system could parse the "modificado por el Artículo X de la Ley Y" part.
                article_num = chunks[-1].article_number if chunks else "1"
                
                # We prepend a clear statement to the text to give the LLM context.
                contextual_text = f"Texto modificado del Artículo {article_num}: {modified_text}"
                
                legal_refs, cross_refs = self.extract_legal_references(contextual_text)

                mod_chunk = LegalChunk(
                    chunk_id=self._generate_chunk_id(pdf_name, page_num, f"{article_num}-mod", i),
                    text=contextual_text,
                    pdf_name=pdf_name, pdf_path=pdf_path, page_number=page_num,
                    chunk_type='structural_modification',
                    document_title=doc_metadata.get('document_title'),
                    document_type=doc_metadata.get('document_type'),
                    document_number=doc_metadata.get('document_number'),
                    promulgation_date=doc_metadata.get('promulgation_date'),
                    chapter_number=current_chapter_info.get('number'),
                    chapter_title=current_chapter_info.get('title'),
                    article_number=article_num,
                    legal_references=legal_refs, cross_references=cross_refs,
                    parent_context=f"Modificación al Artículo {article_num}"
                )
                chunks.append(mod_chunk)

        return chunks

    def _generate_chunk_id(self, pdf_name: str, page_num: int, article_num: str, sub_idx: int) -> str:
        """Generate a unique and deterministic chunk ID."""
        base = f"{pdf_name}_p{page_num}_art{article_num}_s{sub_idx}"
        return hashlib.md5(base.encode()).hexdigest()

class PineconeVectorStore:
    """Pinecone vector storage with advanced operations"""
    
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int = 1536):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        if index_name not in self.pc.list_indexes().names():
            print(f"Creating new Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=environment)
            )
        self.index = self.pc.Index(index_name)
    
    def upsert_chunks(self, chunks: List[LegalChunk], embeddings: np.ndarray, batch_size: int = 100):
        """Upload chunks with embeddings to Pinecone"""
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            vectors = []
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                metadata = {
                    'text': chunk.text[:40000], # Max metadata size for Pinecone
                    'pdf_name': chunk.pdf_name,
                    'page_number': chunk.page_number,
                    'document_title': chunk.document_title or '',
                    'document_type': chunk.document_type or '',
                    'article_number': chunk.article_number or '',
                }
                vectors.append({
                    'id': chunk.chunk_id,
                    'values': embedding.tolist(),
                    'metadata': metadata
                })
            if vectors:
                self.index.upsert(vectors=vectors)

    def search(self, query_embedding: np.ndarray, top_k: int = 10, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search vectors with optional metadata filtering"""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        return results['matches']

class HybridRetriever:
    """Advanced hybrid retrieval combining dense, sparse, and reranking"""
    
    def __init__(self, openai_api_key: Optional[str] = None, cohere_api_key: Optional[str] = None, use_openai: bool = True):
        self.openai_client = None
        if use_openai and openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.embedding_model = "openai"
        else:
            self.embedding_model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
            self.sentence_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_model = "sentence-transformers"
        
        self.bm25 = None
        self.chunks = []
        
        self.cohere_client = cohere.Client(cohere_api_key) if cohere_api_key else None
    
    def get_embedding(self, text: str) -> np.ndarray:
        if self.embedding_model == "openai":
            response = self.openai_client.embeddings.create(model="text-embedding-3-small", input=text.replace("\n", " "))
            return np.array(response.data[0].embedding)
        else:
            return self.sentence_model.encode(text)

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.embedding_model == "openai":
            # OpenAI API handles batching internally, but we can chunk for robustness
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = [t.replace("\n", " ") for t in texts[i:i+batch_size]]
                response = self.openai_client.embeddings.create(model="text-embedding-3-small", input=batch)
                all_embeddings.extend([d.embedding for d in response.data])
            return np.array(all_embeddings)
        else:
            return self.sentence_model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    def initialize_bm25(self, chunks: List[LegalChunk]):
        """Initialize BM25 index from a list of LegalChunk objects."""
        self.chunks = chunks
        if chunks:
            tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Perform BM25 search and return (index, score) tuples."""
        if not self.bm25: return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[int]:
        """Rerank using Cohere and return the new order of indices."""
        if not self.cohere_client or not documents:
            return list(range(min(top_k, len(documents))))
        
        try:
            response = self.cohere_client.rerank(
                model='rerank-multilingual-v2.0',
                query=query, documents=documents, top_n=top_k
            )
            return [result.index for result in response.results]
        except Exception as e:
            print(f"Cohere rerank failed: {e}. Returning original order.")
            return list(range(min(top_k, len(documents))))

    def hybrid_search(self, 
                     query: str,
                     vector_store: PineconeVectorStore,
                     initial_k: int = 10,
                     final_k: int = 7,
                     alpha: float = 0.55) -> List[Dict]:
        """
        Performs a two-stage hybrid search:
        1. Fetches a large candidate pool (initial_k) using a hybrid of dense and sparse scores.
        2. Uses a powerful re-ranker to select the best results (final_k) from the pool.
        """
        if not self.chunks:
            return []

        # 1. Dense (vector) retrieval
        query_embedding = self.get_embedding(query)
        dense_results = vector_store.search(query_embedding, top_k=initial_k)
        
        # 2. Sparse (keyword) retrieval
        sparse_results = self.bm25_search(query, top_k=initial_k)
        
        # 3. Combine scores
        combined_scores = {}
        dense_map = {res['id']: res['score'] for res in dense_results}
        
        for chunk_id, score in dense_map.items():
            combined_scores[chunk_id] = alpha * score
        
        if sparse_results:
            max_sparse_score = max(score for _, score in sparse_results) if sparse_results else 1.0
            for idx, score in sparse_results:
                chunk_id = self.chunks[idx].chunk_id
                normalized_score = score / max_sparse_score
                
                if chunk_id in combined_scores:
                    combined_scores[chunk_id] += (1 - alpha) * normalized_score
                else:
                    combined_scores[chunk_id] = (1 - alpha) * normalized_score
        
        # Get top candidates from the combined scores
        top_candidate_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:initial_k]
        
        # 4. Re-rank the top candidates
        id_to_chunk = {c.chunk_id: c for c in self.chunks}
        ordered_chunks = [id_to_chunk[cid] for cid in top_candidate_ids if cid in id_to_chunk]
        candidate_texts = [c.text for c in ordered_chunks]
        
        if self.cohere_client and candidate_texts:
            reranked_indices = self.rerank(query, candidate_texts, top_k=final_k)
            final_chunks = [ordered_chunks[i] for i in reranked_indices]
        else:
            final_chunks = ordered_chunks[:final_k]

        # Format the final results for the engine
        final_results = [{
            'chunk_id': chunk.chunk_id,
            'chunk': chunk,
            'combined_score': combined_scores.get(chunk.chunk_id, 0.0)
        } for chunk in final_chunks]
        
        return final_results