"""
chunking.py — VeloraTech RAG Pipeline
Stage 2: Chunk

Responsibilities:
  - Split clean text into overlapping character-level windows
  - Attach metadata to every chunk (source, index, char positions)
  - Return chunks ready to be embedded

Nothing in here should know about embeddings or Chroma.

Design decisions:
  - Character-based (not token-based) — model-agnostic, predictable
  - Overlap prevents a sentence from being cut in half at a boundary
  - Paragraph-aware splitting respects natural document structure
"""

import re
import uuid


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[dict]:
    """
    Split text into overlapping character windows.

    Args:
        text:       Clean text to split (output of clean_text()).
        source:     Filename or identifier — stored in every chunk's metadata.
        chunk_size: Target character length per chunk. Default 600.
        overlap:    Number of characters carried from the end of one chunk
                    into the start of the next. Default 100.

    Returns:
        List of chunk dicts:
        {
            "id":          str,   # stable UUID for this chunk
            "text":        str,   # the chunk content
            "source":      str,   # from the document
            "chunk_index": int,   # 0-based position in document
            "char_start":  int,   # character offset in original text
            "char_end":    int,   # character offset end
            "char_count":  int,   # length of this chunk
        }

    Raises:
        ValueError: if chunk_size <= overlap (would produce infinite loop)
        ValueError: if text is empty
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
        )

    chunks = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_content = text[start:end].strip()

        # Skip chunks that are only whitespace (can happen near end of doc)
        if chunk_content:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_content,
                "source": source,
                "chunk_index": index,
                "char_start": start,
                "char_end": end,
                "char_count": len(chunk_content),
            })
            index += 1

        start += step

    return chunks


# ---------------------------------------------------------------------------
# Paragraph-aware chunker (preferred for prose documents)
# ---------------------------------------------------------------------------

def chunk_by_paragraphs(
    text: str,
    source: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[dict]:
    """
    Split text by paragraphs first, then merge/split into target size windows.

    Why: A hard character cut can slice a sentence mid-word. Paragraph
    splitting ensures chunk boundaries always fall between natural breaks.
    Paragraphs shorter than chunk_size are merged; longer ones are split
    with the same sliding window as chunk_text().

    Preferred over chunk_text() for articles, reports, and prose.
    Use chunk_text() for structured data or code snippets.

    Args: same as chunk_text()
    Returns: same schema as chunk_text()
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
        )

    # Split on blank lines (paragraph boundary)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Merge short paragraphs into windows up to chunk_size
    merged_segments = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Flush what we have
            if current:
                merged_segments.append(current)
            # If this single paragraph exceeds chunk_size, hard-split it
            if len(para) > chunk_size:
                merged_segments.extend(_hard_split(para, chunk_size, overlap))
                current = ""
            else:
                current = para

    if current:
        merged_segments.append(current)

    # Build chunk dicts from segments, computing char offsets into original text
    chunks = []
    search_start = 0

    for index, segment in enumerate(merged_segments):
        # Find position in original text (approximate — good enough for metadata)
        pos = text.find(segment[:50], search_start)
        char_start = pos if pos != -1 else search_start
        char_end = char_start + len(segment)
        search_start = max(search_start, char_end - overlap)

        chunks.append({
            "id": str(uuid.uuid4()),
            "text": segment,
            "source": source,
            "chunk_index": index,
            "char_start": char_start,
            "char_end": char_end,
            "char_count": len(segment),
        })

    return chunks


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Internal: split a single oversized string into character windows."""
    parts = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start += step
    return parts


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def inspect_chunks(chunks: list[dict], show_text_preview: int = 120) -> None:
    """
    Print a human-readable summary of a chunk list.
    Use this to verify chunking quality before embedding.

    Args:
        chunks:            Output of chunk_text() or chunk_by_paragraphs()
        show_text_preview: Number of chars to show per chunk (0 = hide text)
    """
    if not chunks:
        print("[chunking] No chunks to inspect.")
        return

    sizes = [c["char_count"] for c in chunks]
    avg = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    print(f"\n{'─' * 60}")
    print(f"[chunking] Total chunks : {len(chunks)}")
    print(f"[chunking] Source       : {chunks[0]['source']}")
    print(f"[chunking] Avg size     : {avg:.0f} chars")
    print(f"[chunking] Min / Max    : {min_size} / {max_size} chars")
    print(f"{'─' * 60}")

    if show_text_preview > 0:
        for c in chunks:
            preview = c["text"][:show_text_preview].replace("\n", " ")
            ellipsis = "…" if c["char_count"] > show_text_preview else ""
            print(f"\n  [{c['chunk_index']:03d}] chars {c['char_start']}–{c['char_end']}")
            print(f"        {preview}{ellipsis}")

    print(f"{'─' * 60}\n")
