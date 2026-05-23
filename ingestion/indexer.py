import json
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch("http://localhost:9200")

print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded.")

INDEX_NAME = "rag-chunks"


def create_index():
    """
    Creates the Elasticsearch index.
    Fixed mappings compatible with ES 8.13.
    """
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")

    mappings = {
        "mappings": {
            "properties": {
                "chunk_id":    {"type": "keyword"},
                "file_name":   {"type": "keyword"},
                "drive_url":   {"type": "keyword"},
                "chunk_index": {"type": "integer"},

                # BM25 full-text search
                "text": {
                    "type": "text",
                    "analyzer": "standard"
                },

                # Dense vector for sentence-transformers
                # all-MiniLM-L6-v2 produces 384 dimensions
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }

    es.indices.create(index=INDEX_NAME, body=mappings)
    print(f"Created index: {INDEX_NAME}")


def encode_text(text):
    """
    Converts text into a 384-dimensional dense vector.
    """
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def index_chunks(chunks_path='data/chunks.json'):
    """
    Reads chunks from JSON, encodes each one,
    and indexes into Elasticsearch.
    """
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)

    print(f"\nIndexing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        vector = encode_text(chunk['text'])

        doc = {
            "chunk_id":     chunk['chunk_id'],
            "file_name":    chunk['file_name'],
            "drive_url":    chunk['drive_url'],
            "chunk_index":  chunk['chunk_index'],
            "text":         chunk['text'],
            "dense_vector": vector
        }

        es.index(
            index=INDEX_NAME,
            id=chunk['chunk_id'],
            document=doc
        )

        print(f"  Indexed {i+1}/{len(chunks)}: {chunk['chunk_id']}")

    es.indices.refresh(index=INDEX_NAME)
    print(f"\nAll {len(chunks)} chunks indexed!")


def verify_index():
    """
    Runs a quick BM25 test search to confirm everything works.
    """
    print("\nVerifying with test search...")

    count = es.count(index=INDEX_NAME)
    print(f"Total documents in index: {count['count']}")

    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match": {
                    "text": "docker container"
                }
            },
            "size": 2
        }
    )

    hits = response['hits']['hits']
    print(f"Test search returned {len(hits)} result(s)")

    for hit in hits:
        print(f"\n  Score: {hit['_score']:.4f}")
        print(f"  File:  {hit['_source']['file_name']}")
        print(f"  Chunk: {hit['_source']['chunk_id']}")
        print(f"  Text:  {hit['_source']['text'][:120]}...")


if __name__ == "__main__":
    print("=== Starting Elasticsearch indexing ===\n")

    print("[1] Creating index...")
    create_index()

    print("\n[2] Indexing chunks...")
    index_chunks()

    print("\n[3] Verifying...")
    verify_index()

    print("\n=== Indexing complete ===")