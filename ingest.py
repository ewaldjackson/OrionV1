"""
ingest.py — VeloraTech RAG Pipeline
Stage 1: Load + Clean

Responsibilities:
  - Read raw .txt files from disk
  - Normalise and clean text before chunking
  - Return clean string + source metadata

Nothing in here should know about chunks, vectors, or Chroma.
"""

import os
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_txt(file_path: str) -> dict:
    """
    Load a .txt file and return its content with source metadata.

    Returns:
        {
            "text": str,          # raw file content
            "source": str,        # filename (no path)
            "file_path": str,     # full resolved path
            "char_count": int     # length before cleaning
        }

    Raises:
        FileNotFoundError: if path does not exist
        ValueError:        if file is empty
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        raise ValueError(f"File is empty: {path}")

    return {
        "text": text,
        "source": path.name,
        "file_path": str(path),
        "char_count": len(text),
    }


def load_directory(dir_path: str, extension: str = ".txt") -> list[dict]:
    """
    Load all files with a given extension from a directory.

    Returns a list of document dicts (same shape as load_txt).
    Logs and skips files that fail to load.
    """
    dir_path = Path(dir_path).resolve()
    results = []

    files = sorted(dir_path.glob(f"*{extension}"))
    if not files:
        print(f"[ingest] No {extension} files found in: {dir_path}")
        return results

    for file in files:
        try:
            doc = load_txt(str(file))
            results.append(doc)
            print(f"[ingest] Loaded: {file.name}  ({doc['char_count']:,} chars)")
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"[ingest] Skipped {file.name}: {e}")

    print(f"[ingest] Total loaded: {len(results)} file(s)")
    return results


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise raw text before chunking.

    Operations (in order):
      1. Unicode normalise to NFC  (merges composed characters)
      2. Remove null bytes and non-printable control characters
         (keep: newlines \n, tabs \t, carriage returns \r)
      3. Collapse runs of spaces/tabs on the same line → single space
      4. Collapse 3+ consecutive blank lines → two newlines (paragraph break)
      5. Strip leading/trailing whitespace on every line
      6. Strip the whole document's leading/trailing whitespace

    Does NOT:
      - Remove punctuation (hurts retrieval)
      - Lowercase (preserves proper nouns)
      - Remove stop words (context matters for embeddings)
    """
    # 1. Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # 2. Strip non-printable control characters (keep \n \r \t)
    text = re.sub(r"[^\S\n\r\t ]+", " ", text)   # replace odd whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Collapse horizontal whitespace (spaces/tabs) on each line
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Collapse 3+ blank lines → paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # 6. Strip document edges
    return text.strip()


def ingest_file(file_path: str) -> dict:
    """
    Full ingest for a single file: load → clean.

    Returns the document dict with an added "clean_text" key
    and "clean_char_count" for comparison/debugging.
    """
    doc = load_txt(file_path)
    doc["clean_text"] = clean_text(doc["text"])
    doc["clean_char_count"] = len(doc["clean_text"])

    reduction = 1 - doc["clean_char_count"] / doc["char_count"]
    print(
        f"[ingest] Cleaned '{doc['source']}': "
        f"{doc['char_count']:,} → {doc['clean_char_count']:,} chars "
        f"({reduction:.1%} reduction)"
    )
    return doc
