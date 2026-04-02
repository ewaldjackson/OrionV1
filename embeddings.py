"""
embeddings.py — VeloraTech RAG Pipeline
Stage 3 + 4: Embed → Store

Responsibilities:
  - Load the embedding model (once, cached on the instance)
  - Convert chunk texts → vectors
  - Persist vectors + text + metadata into a Chroma collection
  - Provide a clean interface for the query stage

Nothing in here should know about raw file loading or chunking logic.

Model choice: all-MiniLM-L6-v2
  - 384 dimensions, ~22MB, runs on CPU in milliseconds
  - Strong semantic quality for its size
  - Good default until you have a reason to switch

Swap by changing MODEL_NAME — the rest of the code is model-agnostic.
"""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from app.config import EMBEDDING_MODEL, CHROMA_PATH

MODEL_NAME          = EMBEDDING_MODEL
DEFAULT_CHROMA_PATH = CHROMA_PATH


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """
    Wraps SentenceTransformer and Chroma into a single pipeline object.

    Usage:
        embedder = Embedder()
        embedder.store_chunks(chunks, collection_name="my_docs")
    """

    def __init__(self, model_name: str = MODEL_NAME, chroma_path: str = DEFAULT_CHROMA_PATH):
        print(f"[embeddings] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chroma_path = chroma_path

        os.makedirs(chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        print(f"[embeddings] Chroma storage: {chroma_path}")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of strings into a list of embedding vectors.

        Args:
            texts: Any list of strings — chunks, queries, whatever.

        Returns:
            List of float vectors, same length as texts.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list.")

        vectors = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity works best normalised
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        Applies the same normalisation as embed() — critical for consistency.

        The query should be cleaned the same way as document text
        (strip extra whitespace, normalise unicode) before calling this.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        vector = self.model.encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.tolist()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """
        Get an existing Chroma collection or create it if absent.
        Collection names must be lowercase alphanumeric + underscores/hyphens.
        """
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},   # cosine distance for retrieval
        )

    def store_chunks(
        self,
        chunks: list[dict],
        collection_name: str = "documents",
    ) -> chromadb.Collection:
        """
        Embed and store a list of chunk dicts into Chroma.

        Args:
            chunks:          Output of chunking.chunk_text() or chunk_by_paragraphs()
            collection_name: Chroma collection to write to. Created if absent.

        Returns:
            The Chroma collection (for inspection or chaining).

        Each chunk is stored with:
            - id:        chunk["id"]          (UUID)
            - embedding: computed vector
            - document:  chunk["text"]        (the raw chunk content)
            - metadata:  source, chunk_index, char_start, char_end, char_count
        """
        if not chunks:
            raise ValueError("No chunks to store.")

        collection = self.get_or_create_collection(collection_name)

        # Extract fields
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
                "char_count": c["char_count"],
            }
            for c in chunks
        ]

        print(f"[embeddings] Embedding {len(texts)} chunks…")
        embeddings = self.embed(texts)

        # Chroma upsert: safe to re-run (updates existing ids, inserts new ones)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        count = collection.count()
        print(
            f"[embeddings] Stored {len(chunks)} chunk(s) → "
            f"'{collection_name}' ({count} total in collection)"
        )
        return collection

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return the names of all collections in this Chroma instance."""
        return [c.name for c in self.client.list_collections()]

    def collection_info(self, collection_name: str) -> dict:
        """Return basic stats about a collection."""
        col = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "count": col.count(),
            "metadata": col.metadata,
        }

    def verify_storage(self, collection_name: str, n_samples: int = 3) -> None:
        """
        Print sample entries from a stored collection.
        Use this after store_chunks() to confirm everything looks right.
        """
        col = self.client.get_collection(collection_name)
        total = col.count()

        if total == 0:
            print(f"[embeddings] Collection '{collection_name}' is empty.")
            return

        sample = col.peek(limit=min(n_samples, total))

        print(f"\n{'─' * 60}")
        print(f"[embeddings] Collection : '{collection_name}'")
        print(f"[embeddings] Total docs : {total}")
        print(f"{'─' * 60}")

        for i, (doc_id, doc_text, meta) in enumerate(
            zip(sample["ids"], sample["documents"], sample["metadatas"])
        ):
            preview = doc_text[:100].replace("\n", " ")
            print(f"\n  Sample {i + 1}")
            print(f"    id      : {doc_id[:16]}…")
            print(f"    source  : {meta['source']}")
            print(f"    index   : {meta['chunk_index']}")
            print(f"    chars   : {meta['char_start']}–{meta['char_end']}")
            print(f"    preview : {preview}…")

        print(f"{'─' * 60}\n")
