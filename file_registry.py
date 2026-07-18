"""
File registry - tracks which PDFs have already been processed into the
knowledge base, keyed by a SHA-256 hash of the file's raw bytes (not just
the filename). This means:

  - Uploading the exact same file again is detected and skipped, even if
    it's re-uploaded under a different name.
  - If a file with the SAME name but DIFFERENT content is uploaded (i.e.
    it changed), the hash is different, so it's correctly treated as new
    and gets (re)processed.

Persisted as a JSON file (processed_files.json) so it survives app
restarts, sitting alongside the FAISS and BM25 indexes.
"""

import os
import json
import hashlib
from datetime import datetime, timezone

from config import REGISTRY_PATH


def compute_file_hash(file_bytes):
    """SHA-256 hash of raw file content - used as the unique file ID."""
    return hashlib.sha256(file_bytes).hexdigest()


def _load_raw():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw(data):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_processed(file_hash):
    """True if this exact file content has already been indexed."""
    return file_hash in _load_raw()


def get_entry(file_hash):
    return _load_raw().get(file_hash)


def register_file(file_hash, filename, subject="General", chunk_count=0, page_count=None):
    """Record a newly-indexed file so it won't be reprocessed next time."""
    data = _load_raw()
    data[file_hash] = {
        "filename": filename,
        "subject": subject,
        "chunk_count": chunk_count,
        "page_count": page_count,
        "upload_date": datetime.now(timezone.utc).isoformat()
    }
    _save_raw(data)


def list_processed_files():
    """All indexed files, oldest first: [{hash, filename, subject, ...}]."""
    data = _load_raw()
    return [
        {"hash": h, **entry}
        for h, entry in sorted(data.items(), key=lambda kv: kv[1].get("upload_date", ""))
    ]


def list_subjects():
    """Unique subjects/categories seen so far - used to populate filter UI."""
    data = _load_raw()
    return sorted({entry.get("subject", "General") for entry in data.values()})


def list_filenames():
    """Unique filenames seen so far - used to populate filter UI."""
    data = _load_raw()
    return sorted({entry.get("filename") for entry in data.values()})


def registry_exists():
    return os.path.exists(REGISTRY_PATH) and len(_load_raw()) > 0