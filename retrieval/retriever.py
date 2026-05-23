from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

INDEX_NAME = "rag-chunks"


def encode_query(query_text):
    """
    Converts the user's question into a 384-dimensional dense vector.
    Same model used at index time — so vectors are comparable.
    """
    vector = model.encode(query_text, normalize_embeddings=True)
    return vector.tolist()


def search_bm25(query_text, top_k=5):
    """
    BM25 keyword search — classic full-text search.
    Elasticsearch does this automatically on the 'text' field.
    Good for exact keyword matches.
    """
    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match": {
                    "text": {
                        "query": query_text
                    }
                }
            },
            "size": top_k
        }
    )
    return _format_hits(response['hits']['hits'])


def search_dense(query_text, top_k=5):
    """
    Dense vector search — semantic similarity search.
    Finds chunks that are semantically similar even if
    they don't share exact keywords with the query.
    """
    query_vector = encode_query(query_text)

    response = es.search(
        index=INDEX_NAME,
        body={
            "knn": {
                "field": "dense_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 50
            },
            "size": top_k
        }
    )
    return _format_hits(response['hits']['hits'])


def search_hybrid(query_text, top_k=5):
    """
    Hybrid search — combines BM25 + dense vector results.
    Uses a two-step approach compatible with ES 8.13:
    1. Get BM25 results
    2. Get dense vector results
    3. Manually combine using Reciprocal Rank Fusion
    """
    bm25_results = search_bm25(query_text, top_k * 2)
    dense_results = search_dense(query_text, top_k * 2)

    # Reciprocal Rank Fusion — combine scores by rank position
    # Formula: score = 1 / (rank + 60) for each list, then sum
    rrf_scores = {}

    for rank, result in enumerate(bm25_results):
        cid = result['chunk_id']
        rrf_scores[cid] = rrf_scores.get(cid, {**result, 'rrf_score': 0.0})
        rrf_scores[cid]['rrf_score'] += 1.0 / (rank + 60)

    for rank, result in enumerate(dense_results):
        cid = result['chunk_id']
        if cid not in rrf_scores:
            rrf_scores[cid] = {**result, 'rrf_score': 0.0}
        rrf_scores[cid]['rrf_score'] += 1.0 / (rank + 60)

    # Sort by combined RRF score descending
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )

    # Return top_k, using rrf_score as the score field
    final = []
    for r in sorted_results[:top_k]:
        final.append({
            'chunk_id':  r['chunk_id'],
            'file_name': r['file_name'],
            'drive_url': r['drive_url'],
            'text':      r['text'],
            'score':     round(r['rrf_score'], 6)
        })

    return final


def retrieve(query_text, mode="hybrid", top_k=5):
    """
    Main retrieval function — called by the API and LLM generator.

    Args:
        query_text: the user's question
        mode: "hybrid" (default) or "bm25"
        top_k: number of chunks to return (default 5)

    Returns:
        List of chunk dicts with text + metadata
    """
    print(f"\nRetrieving with mode='{mode}', top_k={top_k}")
    print(f"Query: {query_text}")

    if mode == "hybrid":
        results = search_hybrid(query_text, top_k)
    elif mode == "bm25":
        results = search_bm25(query_text, top_k)
    elif mode == "dense":
        results = search_dense(query_text, top_k)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'hybrid', 'bm25', or 'dense'")

    print(f"Retrieved {len(results)} chunks")
    return results


def _format_hits(hits):
    """
    Converts raw Elasticsearch hits into clean dicts
    that the rest of the system can use.
    Each result includes the text and all metadata needed for citations.
    """
    results = []
    for hit in hits:
        source = hit['_source']
        results.append({
            'chunk_id':   source['chunk_id'],
            'file_name':  source['file_name'],
            'drive_url':  source['drive_url'],
            'text':       source['text'],
            'score':      hit.get('_score', 0.0)
        })
    return results