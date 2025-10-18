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
from openai import OpenAI


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


def map_topics_to_subjects_llm(client_topics: list[str], available_subjects: list[str]) -> list[str]:
    """
    Uses an OpenAI LLM to map a list of client topics/sectors to a list of
    official, predefined regulatory subjects.

    Args:
        client_topics (list[str]): A list of topics provided by the user.
        available_subjects (list[str]): The official list of subjects to map against.

    Returns:
        list[str]: A deduplicated list of official subjects that match the client topics.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    print(f"Mapping client topics to official subjects using OpenAI...")

    # Format the lists for clear inclusion in the prompt
    client_topics_str = "\n- ".join(client_topics)
    available_subjects_str = "\n- ".join(available_subjects)

    system_prompt = (
        "You are an intelligent legal-domain routing system. Your task is to accurately map a list of "
        "client topics to a predefined list of official regulatory subjects. You must only use subjects "
        "from the official list provided."
    )
    
    user_prompt = (
        f"Here is a list of client topics of interest:\n- {client_topics_str}\n\n"
        f"Here is the complete list of official regulatory subjects:\n- {available_subjects_str}\n\n"
        "For each client topic, identify all relevant official subjects from the list. "
        "A single client topic can map to zero, one, or multiple official subjects. "
        "Your response MUST be a valid JSON object. The JSON object should have a single key, 'mappings', "
        "which is a dictionary. Each key in this dictionary should be a client topic, and its value should be a list "
        "of the corresponding official subject names. If a client topic has no match, its value should be an empty list []."
        "\nDo not include any explanations or conversational text outside of the JSON object."
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0  # For deterministic mapping
        )
        
        response_content = response.choices[0].message.content
        parsed_response = json.loads(response_content)
        
        # --- Validation and Consolidation ---
        # Never trust the LLM's output without validation.
        
        final_matched_subjects = set()
        mappings = parsed_response.get("mappings", {})
        
        for topic, matches in mappings.items():
            for match in matches:
                # Ensure the LLM didn't hallucinate a subject name
                if match in available_subjects:
                    final_matched_subjects.add(match)
                else:
                    print(f"Warning: LLM returned a non-existent subject '{match}' for topic '{topic}'. It will be ignored.")
        
        print(f"  -> LLM successfully mapped to: {list(final_matched_subjects)}")
        return list(final_matched_subjects)

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from OpenAI response: {response_content}")
        return []
    except Exception as e:
        print(f"An error occurred while communicating with OpenAI: {e}")
        # Re-raise as an HTTPException so the endpoint can handle it
        raise Exception(status_code=503, detail=f"OpenAI API error: {e}")



if __name__ == "__main__":
    # Example workflow:
    # 1. manage_version_rotation("versions/new")
    # 2. process_regulatory_changes()
    manage_version_rotation("versions/new")
    process_regulatory_changes()
