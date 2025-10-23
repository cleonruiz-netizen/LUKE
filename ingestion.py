"""
LUKE Regulatory Ingestion Pipeline (Supabase-integrated + Cloud-persistent)
---------------------------------------------------------------------------
Pulls the latest PDFs directly from Supabase, compares them against hashes,
chunks and embeds only new/modified files, upserts to Pinecone, and stores
the local index & logs back in Supabase Storage.
"""

import os
import io
import json
import hashlib
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import fitz
from dataclasses import asdict
from dotenv import load_dotenv
from supabase import create_client, Client

from core.components import EnhancedLegalChunker, PineconeVectorStore, HybridRetriever
import config

load_dotenv()

# --- Supabase Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET]):
    raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

LATEST_DIR = "versions/latest"
INDEX_DIR = "indexes"  # Folder in Supabase where we store index + log


# ====================== HELPER FUNCTIONS ======================

def get_subjects_from_supabase():
    """List all subject folders under the latest version."""
    try:
        items = supabase.storage.from_(SUPABASE_BUCKET).list(LATEST_DIR)
        return [item["name"] for item in items if "." not in item["name"]]
    except Exception as e:
        print(f"✗ Could not list subjects from Supabase: {e}")
        return []


def download_hashes(subject: str) -> dict:
    """Download the hashes.json for a specific subject."""
    try:
        data = supabase.storage.from_(SUPABASE_BUCKET).download(f"{LATEST_DIR}/{subject}/hashes.json")
        return json.loads(data.decode("utf-8"))
    except Exception:
        print(f"⚠ No hashes.json found for {subject}.")
        return {}


def download_pdf(subject: str, filename: str) -> bytes:
    """Download a PDF file from Supabase."""
    try:
        return supabase.storage.from_(SUPABASE_BUCKET).download(f"{LATEST_DIR}/{subject}/{filename}")
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return None


# ====================== INDEX / LOG MANAGEMENT ======================

def load_from_supabase(filename: str) -> dict:
    """Download and load a JSON file from Supabase if it exists."""
    try:
        data = supabase.storage.from_(SUPABASE_BUCKET).download(f"{INDEX_DIR}/{filename}")
        print(f"☁️ Loaded {filename} from Supabase.")
        return json.loads(data.decode("utf-8"))
    except Exception:
        print(f"⚠ No remote {filename} found. Starting fresh.")
        return {}


def upload_to_supabase(filename: str, data: dict):
    """Upload a JSON file to Supabase (upsert)."""
    try:
        json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=f"{INDEX_DIR}/{filename}",
            file=json_bytes,  # pass bytes, not BytesIO
            file_options={"content_type": "application/json", "upsert": True}
        )
        print(f"☁️ Uploaded {filename} to Supabase successfully.")
    except Exception as e:
        print(f"✗ Failed to upload {filename} to Supabase: {e}")


def get_file_hash(file_bytes: bytes) -> str:
    """Compute SHA256 hash for a given PDF (in-memory)."""
    return hashlib.sha256(file_bytes).hexdigest()


def process_pdf_worker(args):
    """Process a single PDF in a worker process (extract + chunk)."""
    subject, filename, pdf_bytes = args
    chunker = EnhancedLegalChunker()

    try:
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)
        if not full_text.strip():
            return []

        doc_metadata = chunker.extract_document_metadata(full_text, filename)
        all_chunks = []
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if page_text.strip():
                    page_chunks = chunker.create_contextual_chunks(
                        page_text, filename, f"{subject}/{filename}", page_num, doc_metadata
                    )
                    all_chunks.extend(page_chunks)
        return all_chunks
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")
        return None


# ====================== MAIN INGESTION ======================

def ingest_subject(subject: str, processed_log: dict, local_index: dict):
    """Ingest all new or changed PDFs for a single subject."""
    print(f"\n{'='*70}\nStarting ingestion for subject: {subject}\n{'='*70}")

    hashes_data = download_hashes(subject)
    if not hashes_data or "files" not in hashes_data:
        print(f"⚠ No valid hashes.json found for {subject}. Skipping.")
        return processed_log, local_index

    latest_files = hashes_data["files"]
    files_to_process = []

    # Determine which PDFs are new or changed
    for filename, meta in latest_files.items():
        remote_hash = meta.get("hash")
        key = f"{subject}/{filename}"
        if processed_log.get(key) != remote_hash:
            files_to_process.append((subject, filename))
            processed_log[key] = remote_hash  # optimistic update

    if not files_to_process:
        print(f"✓ No new or modified PDFs for {subject}. Index is up to date.")
        return processed_log, local_index

    print(f"Found {len(files_to_process)} new or modified PDFs in '{subject}'.")

    # --- Download and process PDFs ---
    pdf_args = []
    for subject, filename in files_to_process:
        pdf_bytes = download_pdf(subject, filename)
        if pdf_bytes:
            pdf_args.append((subject, filename, pdf_bytes))

    print(f"Processing {len(pdf_args)} PDFs in parallel...")

    new_chunks = []
    with Pool(processes=cpu_count()) as pool:
        for chunks in tqdm(pool.imap_unordered(process_pdf_worker, pdf_args), total=len(pdf_args)):
            if chunks:
                new_chunks.extend(chunks)

    if not new_chunks:
        print("⚠ No new chunks generated.")
        return processed_log, local_index

    print(f"✓ Total new chunks generated: {len(new_chunks)}")

    # --- Embedding and upsert ---
    retriever = HybridRetriever(
        openai_api_key=config.OPENAI_API_KEY,
        use_openai=config.USE_OPENAI_EMBEDDINGS
    )
    vector_store = PineconeVectorStore(
        api_key=config.PINECONE_API_KEY,
        environment=config.PINECONE_ENVIRONMENT,
        index_name=config.PINECONE_INDEX_NAME
    )

    print("\nEmbedding new chunks...")
    texts = [chunk.text for chunk in new_chunks]
    embeddings = retriever.batch_embed(texts)

    print("Upserting to Pinecone...")
    vector_store.upsert_chunks(new_chunks, embeddings)

    # --- Update BM25 Local Index (in-memory) ---
    existing_chunks = local_index.get("chunks", [])
    chunk_map = {c["chunk_id"]: c for c in existing_chunks}
    for c in new_chunks:
        chunk_map[c.chunk_id] = asdict(c)

    local_index = {
        "chunks": list(chunk_map.values()),
        "timestamp": datetime.now().isoformat()
    }

    print(f"✓ Ingestion complete for {subject}.")
    return processed_log, local_index


def main():
    """Main entrypoint: process all subjects under versions/latest."""
    subjects = get_subjects_from_supabase()
    if not subjects:
        print("✗ No subjects found under 'versions/latest/'.")
        return

    # --- Load state from Supabase ---
    processed_log = load_from_supabase("processed_files_log.json")
    local_index = load_from_supabase("local_index.json")

    print(f"Detected subjects: {subjects}")
    for subject in subjects:
        processed_log, local_index = ingest_subject(subject, processed_log, local_index)

    # --- Upload results back to Supabase ---
    upload_to_supabase("local_index.json", local_index)
    upload_to_supabase("processed_files_log.json", processed_log)
    print("✅ All ingestion data saved to Supabase.")


if __name__ == "__main__":
    main()
