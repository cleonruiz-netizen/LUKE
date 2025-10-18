# versioning.py

"""
LUKE Regulatory Versioning (Supabase Version) - Final Integration with SPIJ Scraper
-----------------------------------------------------------------------------------
Compares and rotates versions of Peruvian legal PDFs stored in Supabase, where each
subject folder contains:
  - PDF files (e.g., law123.pdf)
  - hashes.json  (with SHA256 hashes for every PDF)

Detects new, modified, and deleted documents and uploads human-readable change
summaries to Supabase for later use in Flow 5 (Regulatory Alerts) and Flow 10 (Continuous Improvement).
"""

import os
import io
import json
from supabase import create_client, Client
from dotenv import load_dotenv
from core.versioncompare_engine import VersionCompareEngine

load_dotenv()

# --- Supabase Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET]):
    raise ValueError("Missing Supabase credentials in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Folder Naming Convention in Supabase ---
LATEST_DIR = "versions/latest"
PREVIOUS_DIR = "versions/previous"
CHANGES_DIR = "versions/changes"


# ====================== UTILITIES ======================

def download_json_from_supabase(path: str) -> dict:
    """Download and decode JSON (hashes.json)."""
    try:
        data = supabase.storage.from_(SUPABASE_BUCKET).download(path)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}


def download_pdf_from_supabase(path: str) -> bytes:
    """Download a PDF file as bytes."""
    try:
        return supabase.storage.from_(SUPABASE_BUCKET).download(path)
    except Exception as e:
        print(f"✗ Error downloading {path}: {e}")
        return None


def upload_summary_to_supabase(subject: str, filename: str, content: str):
    """Upload textual change summary to Supabase."""
    summary_path = f"{CHANGES_DIR}/{subject}/{filename}.txt"
    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            summary_path,
            content.encode("utf-8"),
            {"content-type": "text/plain", "upsert": "true"}
        )
        print(f"  ✓ Uploaded change summary: {summary_path}")
    except Exception as e:
        print(f"✗ Failed to upload summary for {filename}: {e}")


def compare_docs_with_engine(latest_bytes: bytes, previous_bytes: bytes, filename: str) -> str:
    """Run document comparison using VersionCompareEngine (in-memory)."""
    try:
        engine = VersionCompareEngine(
            file_v1=io.BytesIO(previous_bytes),
            file_v2=io.BytesIO(latest_bytes)
        )
        response = engine.compare()
        if not response.change_log:
            return f"Document '{filename}' modified, but no substantive differences found."
        summary_parts = [f"Detected changes for document '{filename}':"]
        for change in response.change_log:
            summary_parts.append(f"- {change}")
        return "\n".join(summary_parts)
    except Exception as e:
        return f"Error comparing '{filename}': {e}"


def list_subjects(base_dir: str) -> list[str]:
    """List subfolders (subjects) inside a given directory."""
    try:
        items = supabase.storage.from_(SUPABASE_BUCKET).list(base_dir)
        return [item["name"] for item in items if "." not in item["name"]]
    except Exception:
        return []


# ====================== VERSION COMPARISON ======================

def process_regulatory_changes():
    """Compare latest and previous Supabase versions using subject-level hashes."""
    print("\n" + "=" * 80)
    print("RUNNING SUPABASE CHANGE DETECTION (Integrated with SPIJ Scraper)")
    print("=" * 80)

    latest_subjects = list_subjects(LATEST_DIR)
    previous_subjects = list_subjects(PREVIOUS_DIR)
    all_subjects = set(latest_subjects) | set(previous_subjects)

    for subject in all_subjects:
        print(f"\n--- Processing subject: {subject} ---")

        # Check if subject exists in latest
        if subject not in latest_subjects:
            print(f"    ⚠ Subject '{subject}' no longer exists in latest version (deleted)")
            continue

        latest_hash_path = f"{LATEST_DIR}/{subject}/hashes.json"
        previous_hash_path = f"{PREVIOUS_DIR}/{subject}/hashes.json"

        latest_hashes = download_json_from_supabase(latest_hash_path)
        
        # Handle new subjects (no previous version)
        if subject not in previous_subjects:
            print(f"    ✓ New subject detected: '{subject}'")
            previous_hashes = {"files": {}}
        else:
            previous_hashes = download_json_from_supabase(previous_hash_path)

        latest_files = latest_hashes.get("files", {})
        previous_files = previous_hashes.get("files", {})

        # Detect new or modified PDFs
        for filename, meta in latest_files.items():
            latest_hash = meta.get("hash")
            previous_hash = previous_files.get(filename, {}).get("hash")

            if latest_hash != previous_hash:
                latest_pdf_path = f"{LATEST_DIR}/{subject}/{filename}"
                previous_pdf_path = f"{PREVIOUS_DIR}/{subject}/{filename}" if previous_hash else None

                latest_bytes = download_pdf_from_supabase(latest_pdf_path)
                previous_bytes = download_pdf_from_supabase(previous_pdf_path) if previous_pdf_path else None

                if latest_bytes:
                    # Handle new files (no previous version)
                    if previous_bytes is None:
                        summary = f"New document '{filename}' added to subject '{subject}'."
                    else:
                        summary = compare_docs_with_engine(latest_bytes, previous_bytes, filename)
                    upload_summary_to_supabase(subject, filename, summary)
            else:
                print(f"    ✓ No change detected in '{filename}'")

        # Detect deleted PDFs
        for filename in previous_files:
            if filename not in latest_files:
                summary = f"Document '{filename}' appears to have been removed from subject '{subject}'."
                upload_summary_to_supabase(subject, filename + "_deleted", summary)

    print("\n" + "=" * 80)
    print("SUPABASE CHANGE DETECTION COMPLETE")
    print("=" * 80)


# ====================== VERSION ROTATION ======================

def manage_version_rotation(new_folder_prefix: str):
    """
    Rotate Supabase versions after a new scrape:
      1. Delete old 'previous'
      2. Copy 'latest' → 'previous'
      3. Copy new folder → 'latest' (including hashes.json)
    """
    print(f"\n--- Performing Supabase version rotation for {new_folder_prefix} ---")
    
    # Get all subjects from new folder
    new_subjects = list_subjects(new_folder_prefix)
    
    for subject in new_subjects:
        print(f"\n  Processing subject: {subject}")
        
        try:
            # Step 1: Clear previous version for this subject
            try:
                prev_files = supabase.storage.from_(SUPABASE_BUCKET).list(f"{PREVIOUS_DIR}/{subject}")
                for f in prev_files:
                    supabase.storage.from_(SUPABASE_BUCKET).remove([f"{PREVIOUS_DIR}/{subject}/{f['name']}"])
                print(f"    ✓ Cleared previous/{subject}")
            except Exception:
                print(f"    ⚠ No previous version for {subject}")
            
            # Step 2: Copy latest → previous for this subject
            try:
                latest_files = supabase.storage.from_(SUPABASE_BUCKET).list(f"{LATEST_DIR}/{subject}")
                for f in latest_files:
                    src = f"{LATEST_DIR}/{subject}/{f['name']}"
                    dst = f"{PREVIOUS_DIR}/{subject}/{f['name']}"
                    data = download_pdf_from_supabase(src)
                    if data:
                        content_type = "application/json" if f["name"].endswith(".json") else "application/pdf"
                        supabase.storage.from_(SUPABASE_BUCKET).upload(
                            dst, data, {"content-type": content_type, "upsert": "true"}
                        )
                print(f"    ✓ Promoted latest/{subject} → previous/{subject}")
            except Exception as e:
                print(f"    ⚠ No latest version for {subject}: {e}")
            
            # Step 3: Promote new folder → latest for this subject
            new_files = supabase.storage.from_(SUPABASE_BUCKET).list(f"{new_folder_prefix}/{subject}")
            for f in new_files:
                src = f"{new_folder_prefix}/{subject}/{f['name']}"
                dst = f"{LATEST_DIR}/{subject}/{f['name']}"
                data = download_pdf_from_supabase(src)
                if data:
                    content_type = "application/json" if f["name"].endswith(".json") else "application/pdf"
                    supabase.storage.from_(SUPABASE_BUCKET).upload(
                        dst, data, {"content-type": content_type, "upsert": "true"}
                    )
            print(f"    ✓ Promoted new/{subject} → latest/{subject}")
            
        except Exception as e:
            print(f"    ✗ Error rotating {subject}: {e}")
    
    print("\n--- Supabase version rotation finished successfully. ---")
    # Step 4: Clean up 'versions/new' ONLY after all subjects are promoted
    print("\n--- Cleaning up versions/new after successful promotion ---")
    try:
        cleanup_subjects = list_subjects(new_folder_prefix)
        for subject in cleanup_subjects:
            try:
                cleanup_files = supabase.storage.from_(SUPABASE_BUCKET).list(f"{new_folder_prefix}/{subject}")
                if cleanup_files:
                    files_to_remove = [f"{new_folder_prefix}/{subject}/{f['name']}" for f in cleanup_files]
                    # Remove files in batches (Supabase has limits)
                    batch_size = 100
                    for i in range(0, len(files_to_remove), batch_size):
                        batch = files_to_remove[i:i + batch_size]
                        supabase.storage.from_(SUPABASE_BUCKET).remove(batch)
                    print(f"    ✓ Cleaned up {new_folder_prefix}/{subject} ({len(cleanup_files)} files)")
                else:
                    print(f"    ⚠ No files to clean up in {new_folder_prefix}/{subject}")
            except Exception as e:
                print(f"    ✗ Error cleaning up {subject}: {e}")
        print("✓ All new files cleaned up successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not clean up {new_folder_prefix}: {e}")


if __name__ == "__main__":
    # Example workflow:
    # 1. manage_version_rotation("versions/new")
    # 2. process_regulatory_changes()
    manage_version_rotation("versions/new")
    process_regulatory_changes()
