"""
config.py — VeloraTech RAG Pipeline
Central configuration for all pipeline settings.

All tunable values live here. No magic numbers in other files.
To adjust the pipeline, change values here only.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_PATH   = str(PROJECT_ROOT / "data" / "chroma")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 600   # target characters per chunk
CHUNK_OVERLAP = 100   # characters carried into the next chunk

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Swap to a larger model later without touching any other file:
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ---------------------------------------------------------------------------
# Chroma / retrieval
# ---------------------------------------------------------------------------

COLLECTION_NAME = "documents"
TOP_K           = 3   # number of chunks returned per query
