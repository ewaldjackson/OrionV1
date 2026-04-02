"""
query.py — VeloraTech RAG Pipeline
Stage 5: Retrieve

Responsibilities:
  - Accept a natural language query
  - Clean + embed it (using the same model as ingestion)
  - Query Chroma for the top-K most similar chunks
  - Return structured results for inspection or downstream generation

Nothing in here should write to disk, modify collections, or generate answers.
The retriever's only job is to find the right context.

Key principle:
  The query MUST be cleaned and embedded with the same model and
  normalisation settings used during ingestion. Any mismatch silently
  degrades retrieval quality.
"""

import re
import unicodedata

from app.embeddings import Embedder
from app.config import TOP_K as DEFAULT_TOP_K


# ---------------------------------------------------------------------------
# Query cleaner (mirrors ingest.clean_text — keep in sync)
# ---------------------------------------------------------------------------

def clean_query(query: str) -> str:
    """
    Apply the same normalisation to the query that was applied to documents.

    If ingest.clean_text() changes, update this function too.
    Asymmetric cleaning is one of the most common silent RAG failures.
    """
    query = unicodedata.normalize("NFC", query)
    query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    query = re.sub(r"[ \t]+", " ", query)
    return query.strip()


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Query interface for a stored Chroma collection.

    Usage:
        retriever = Retriever()
        results = retriever.retrieve("What is photosynthesis?", top_k=3)
    """

    def __init__(self, embedder: Embedder = None):
        """
        Args:
            embedder: An Embedder instance. If None, a default one is created.
                      Pass in the same Embedder used for ingestion to share
                      the model cache and avoid loading it twice.
        """
        self.embedder = embedder or Embedder()

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        collection_name: str = "documents",
        top_k: int = DEFAULT_TOP_K,
        where: dict = None,
    ) -> list[dict]:
        """
        Retrieve the top-K most relevant chunks for a query.

        Args:
            query:           Natural language question or phrase.
            collection_name: Which Chroma collection to search.
            top_k:           Number of results to return. Default 3.
            where:           Optional Chroma metadata filter, e.g.
                             {"source": "my_doc.txt"} to restrict to one file.

        Returns:
            List of result dicts, ordered by relevance (most relevant first):
            {
                "rank":       int,   # 1 = best match
                "text":       str,   # the chunk content
                "source":     str,   # filename
                "chunk_index":int,
                "char_start": int,
                "char_end":   int,
                "distance":   float, # cosine distance (lower = more similar)
                "score":      float, # 1 - distance (higher = more similar)
                "id":         str,
            }

        Raises:
            ValueError: if query is empty
            Exception:  if collection does not exist
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        clean = clean_query(query)
        query_vector = self.embedder.embed_query(clean)

        collection = self.embedder.client.get_collection(collection_name)

        kwargs = dict(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw = collection.query(**kwargs)

        results = []
        for rank, (doc_id, text, meta, distance) in enumerate(
            zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            ),
            start=1,
        ):
            results.append({
                "rank": rank,
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "char_start": meta.get("char_start", 0),
                "char_end": meta.get("char_end", 0),
                "distance": round(distance, 4),
                "score": round(1 - distance, 4),
                "id": doc_id,
            })

        return results

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def print_results(self, results: list[dict], show_full_text: bool = False) -> None:
        """
        Pretty-print retrieval results for terminal inspection.

        Args:
            results:        Output of retrieve()
            show_full_text: If True, print the full chunk text.
                            If False, print a 200-char preview.
        """
        if not results:
            print("[query] No results returned.")
            return

        print(f"\n{'═' * 60}")
        print(f"[query] {len(results)} result(s) returned")
        print(f"{'═' * 60}")

        for r in results:
            preview_len = None if show_full_text else 200
            text_display = (
                r["text"] if show_full_text
                else r["text"][:200].replace("\n", " ") + (
                    "…" if len(r["text"]) > 200 else ""
                )
            )

            print(f"\n  Rank #{r['rank']}  |  score: {r['score']:.4f}  |  distance: {r['distance']:.4f}")
            print(f"  Source : {r['source']}  (chunk {r['chunk_index']}, chars {r['char_start']}–{r['char_end']})")
            print(f"  {'─' * 54}")
            print(f"  {text_display}")

        print(f"\n{'═' * 60}\n")

    def score_summary(self, results: list[dict]) -> dict:
        """
        Return a quick stats summary of a result set.
        Useful for comparing retrieval quality across chunking strategies.
        """
        if not results:
            return {}
        scores = [r["score"] for r in results]
        return {
            "top_score": scores[0],
            "mean_score": round(sum(scores) / len(scores), 4),
            "min_score": scores[-1],
            "spread": round(scores[0] - scores[-1], 4),
        }
