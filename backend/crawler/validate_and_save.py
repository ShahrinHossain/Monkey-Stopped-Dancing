import json
import os
from typing import Dict, Any, Optional
from langdetect import detect, LangDetectException


def to_record(doc: Dict[str, Any], expected_language: str) -> Optional[Dict[str, Any]]:
    """
    Validates and converts a document dict to a standardized record.
    
    Args:
        doc: Dictionary with keys: url, title, body, date (optional)
        expected_language: Expected language code ('bn' or 'en')
    
    Returns:
        Standardized record dict or None if validation fails
    """
    if not doc:
        return None
    
    url = doc.get("url", "").strip()
    title = doc.get("title", "").strip()
    body = doc.get("body", "").strip()
    date = doc.get("date")
    
    # Basic validation
    if not url or not title or not body:
        return None
    
    # Minimum body length check
    if len(body) < 200:
        return None
    
    # Language detection/validation
    detected_language = expected_language
    try:
        text_sample = f"{title} {body[:500]}"
        detected = detect(text_sample)
        if detected in ("bn", "bg"):
            detected_language = "bn"
        elif detected == "en":
            detected_language = "en"
        else:
            detected_language = expected_language
    except (LangDetectException, Exception):
        detected_language = expected_language
    
    # Enforce expected language
    if expected_language == "bn" and detected_language != "bn":
        return None
    if expected_language == "en" and detected_language != "en":
        return None
    
    # Count tokens (simple word count)
    tokens_count = len(body.split()) + len(title.split())
    
    record = {
        "title": title,
        "body": body,
        "url": url,
        "date": date,
        "language": expected_language,
        "tokens_count": tokens_count,
    }
    
    return record


def append_jsonl(file_path: str, record: Dict[str, Any]) -> None:
    """
    Appends a record to a JSONL file.
    Creates the file and directory if they don't exist.
    
    Args:
        file_path: Path to the JSONL file
        record: Dictionary to append as a JSON line
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "a", encoding="utf-8") as f:
        json_line = json.dumps(record, ensure_ascii=False)
        f.write(json_line + "\n")
