"""
tests/test_pipeline.py — VeloraTech RAG Pipeline
End-to-end validation: Load → Clean → Chunk → Embed → Store → Retrieve

Run from the project root:
    python -m tests.test_pipeline

What this tests:
  1. Ingest a sample .txt document
  2. Inspect chunks (visual check)
  3. Embed and store in Chroma
  4. Verify storage (sample peek)
  5. Run test queries
  6. Confirm relevant chunks are returned

This is your first milestone proof:
  "The system can take raw data, process it properly,
   and retrieve the correct context for a query."
"""

import sys
import os
from pathlib import Path

# Allow running from project root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import ingest_file
from app.chunking import chunk_by_paragraphs, inspect_chunks
from app.embeddings import Embedder
from app.query import Retriever
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME


# ---------------------------------------------------------------------------
# Sample document (written inline so the test is self-contained)
# ---------------------------------------------------------------------------

SAMPLE_CONTENT = """
The Science of Photosynthesis

Photosynthesis is the process by which green plants, algae, and some bacteria
convert light energy into chemical energy stored as glucose. It is one of the
most fundamental biological processes on Earth, forming the base of nearly
all food chains.

The overall chemical equation for photosynthesis is:
6CO2 + 6H2O + light energy → C6H12O6 + 6O2

This means that carbon dioxide and water, in the presence of sunlight, are
transformed into glucose and oxygen. The glucose produced is used by the
plant as an energy source for growth, repair, and reproduction.

Light-Dependent Reactions

The first stage of photosynthesis occurs in the thylakoid membranes of the
chloroplast. Here, chlorophyll and other pigments absorb sunlight. This
energy is used to split water molecules, releasing oxygen as a byproduct.
The energy is also captured in the form of ATP and NADPH, which are
energy-carrier molecules.

The Calvin Cycle

The second stage, known as the Calvin Cycle or light-independent reactions,
takes place in the stroma of the chloroplast. Here, the ATP and NADPH
produced in the light-dependent reactions are used to fix carbon dioxide
from the atmosphere into organic molecules. This process is also called
carbon fixation. The end product is a three-carbon molecule called G3P
(glyceraldehyde-3-phosphate), which can be used to build glucose and
other organic compounds.

Factors Affecting Photosynthesis

Several environmental factors influence the rate of photosynthesis:

Light intensity plays a major role. As light intensity increases, the
rate of photosynthesis increases up to a saturation point, beyond which
no further increase occurs.

Carbon dioxide concentration is another limiting factor. Higher CO2
levels generally increase the rate of photosynthesis, which is why
some greenhouses enrich their air with CO2.

Temperature also affects the process. Photosynthesis relies on enzymes,
which work best within a specific temperature range — typically between
25°C and 35°C for most plants. Temperatures above or below this range
reduce enzyme efficiency and slow the process.

Water availability is critical. Water is a direct reactant in the
light-dependent reactions. Water stress causes stomata to close,
reducing the entry of CO2 and slowing photosynthesis.

Importance to Life on Earth

Photosynthesis is essential for life on Earth for two reasons. First,
it produces oxygen, which most living organisms require for cellular
respiration. Second, it forms the primary source of organic material
and energy in most ecosystems through the production of glucose.

Without photosynthesis, the oxygen in Earth's atmosphere would be
depleted within a few thousand years, and the base of almost every
food chain would collapse.
""".strip()

SAMPLE_FILE = Path(__file__).parent.parent / "data" / "raw" / "photosynthesis.txt"
COLLECTION_NAME = "test_pipeline"

# Test queries with expected keywords in the correct chunk
TEST_QUERIES = [
    {
        "query": "What happens in the Calvin Cycle?",
        "expect_keyword": "Calvin",
    },
    {
        "query": "How does light intensity affect photosynthesis?",
        "expect_keyword": "light intensity",
    },
    {
        "query": "Why is photosynthesis important for life on Earth?",
        "expect_keyword": "oxygen",
    },
    {
        "query": "What is the chemical equation for photosynthesis?",
        "expect_keyword": "CO2",
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_pipeline_test():
    print("\n" + "=" * 60)
    print("  VeloraTech RAG Pipeline — End-to-End Test")
    print("=" * 60)

    passed = 0
    failed = 0

    # ------------------------------------------------------------------
    # 0. Write sample file
    # ------------------------------------------------------------------
    print("\n[test] Stage 0: Writing sample document…")
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_FILE.write_text(SAMPLE_CONTENT, encoding="utf-8")
    print(f"[test] Written: {SAMPLE_FILE}")

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    print("\n[test] Stage 1: Ingest (Load + Clean)…")
    doc = ingest_file(str(SAMPLE_FILE))
    assert doc["clean_text"], "clean_text must not be empty"
    assert doc["source"] == "photosynthesis.txt"
    print(f"[test] ✓ Ingest passed — {doc['clean_char_count']:,} clean chars")

    # ------------------------------------------------------------------
    # 2. Chunk
    # ------------------------------------------------------------------
    print("\n[test] Stage 2: Chunk…")
    chunks = chunk_by_paragraphs(
        text=doc["clean_text"],
        source=doc["source"],
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    assert len(chunks) > 0, "Must produce at least one chunk"
    assert all(c["text"] for c in chunks), "No chunk should be empty"
    assert all(c["source"] == "photosynthesis.txt" for c in chunks), "Source must propagate"

    inspect_chunks(chunks, show_text_preview=120)
    print(f"[test] ✓ Chunking passed — {len(chunks)} chunks produced")

    # ------------------------------------------------------------------
    # 3. Embed + Store
    # ------------------------------------------------------------------
    print("\n[test] Stage 3: Embed + Store…")
    embedder = Embedder()

    # Clean up previous test collection if it exists
    try:
        embedder.client.delete_collection(COLLECTION_NAME)
        print(f"[test] Cleared previous '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = embedder.store_chunks(chunks, collection_name=COLLECTION_NAME)
    assert collection.count() == len(chunks), (
        f"Stored count {collection.count()} ≠ chunk count {len(chunks)}"
    )

    embedder.verify_storage(COLLECTION_NAME, n_samples=2)
    print(f"[test] ✓ Storage passed — {collection.count()} vectors in Chroma")

    # ------------------------------------------------------------------
    # 4. Retrieve + Validate
    # ------------------------------------------------------------------
    print("\n[test] Stage 4: Retrieve + Validate…")
    retriever = Retriever(embedder=embedder)

    for case in TEST_QUERIES:
        query = case["query"]
        keyword = case["expect_keyword"]

        results = retriever.retrieve(
            query=query,
            collection_name=COLLECTION_NAME,
            top_k=3,
        )

        retriever.print_results(results)
        stats = retriever.score_summary(results)
        print(f"[test] Score summary: {stats}")

        # Validation: the expected keyword must appear in at least one top-3 result
        combined_text = " ".join(r["text"].lower() for r in results)
        if keyword.lower() in combined_text:
            print(f"[test] ✓ PASS — '{keyword}' found in top-3 results for: \"{query}\"")
            passed += 1
        else:
            print(f"[test] ✗ FAIL — '{keyword}' NOT found in top-3 results for: \"{query}\"")
            failed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{passed + failed} queries passed")
    if failed == 0:
        print("  ✓ Pipeline milestone ACHIEVED.")
        print("  The system ingests, stores, and retrieves correctly.")
    else:
        print(f"  ✗ {failed} query/queries failed. Review chunking + retrieval.")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_pipeline_test()
    sys.exit(0 if success else 1)
