from retrieval.retriever import retrieve
import json

def run_tests():
    print("=" * 50)
    print("RETRIEVAL TEST")
    print("=" * 50)

    # Test query — change this to match your PDF content
    query = "What is Docker and how does it work?"

    # --- Test 1: BM25 only ---
    print("\n[TEST 1] BM25 keyword search")
    print("-" * 40)
    results_bm25 = retrieve(query, mode="bm25", top_k=3)
    for i, r in enumerate(results_bm25):
        print(f"\nResult {i+1}:")
        print(f"  File:  {r['file_name']}")
        print(f"  Chunk: {r['chunk_id']}")
        print(f"  Score: {r['score']}")
        print(f"  Text:  {r['text'][:150]}...")

    # --- Test 2: Dense vector only ---
    print("\n[TEST 2] Dense vector semantic search")
    print("-" * 40)
    results_dense = retrieve(query, mode="dense", top_k=3)
    for i, r in enumerate(results_dense):
        print(f"\nResult {i+1}:")
        print(f"  File:  {r['file_name']}")
        print(f"  Chunk: {r['chunk_id']}")
        print(f"  Score: {r['score']}")
        print(f"  Text:  {r['text'][:150]}...")

    # --- Test 3: Hybrid RRF ---
    print("\n[TEST 3] Hybrid search (BM25 + Dense + RRF)")
    print("-" * 40)
    results_hybrid = retrieve(query, mode="hybrid", top_k=3)
    for i, r in enumerate(results_hybrid):
        print(f"\nResult {i+1}:")
        print(f"  File:  {r['file_name']}")
        print(f"  Chunk: {r['chunk_id']}")
        print(f"  Score: {r['score']}")
        print(f"  Text:  {r['text'][:150]}...")

    print("\n" + "=" * 50)
    print("All retrieval tests complete!")
    print("=" * 50)

if __name__ == "__main__":
    run_tests()