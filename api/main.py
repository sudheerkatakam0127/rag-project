from fastapi import FastAPI, HTTPException
from api.models import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    HealthResponse, Citation
)
from llm.generator import query_rag
from ingestion.ingest import run_ingestion
from elasticsearch import Elasticsearch
import ollama
import os

# Create the FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation over PDF documents",
    version="1.0.0"
)

# Elasticsearch client for health check
es = Elasticsearch("http://localhost:9200")


# ─────────────────────────────────────────
# GET /healthz
# ─────────────────────────────────────────
@app.get("/healthz", response_model=HealthResponse)
def health_check():
    """
    Checks that both Elasticsearch and Ollama are reachable.
    Returns their status so you know everything is running.
    """
    # Check Elasticsearch
    try:
        es.ping()
        es_status = "ok"
    except Exception:
        es_status = "unreachable"

    # Check Ollama
    try:
        ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": "ping"}]
        )
        ollama_status = "ok"
    except Exception:
        ollama_status = "unreachable"

    overall = "ok" if es_status == "ok" and ollama_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        elasticsearch=es_status,
        ollama=ollama_status
    )


# ─────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main RAG endpoint.
    Accepts a question, retrieves relevant chunks,
    generates a grounded answer, returns citations.

    Example request body:
    {
        "question": "What is Docker?",
        "mode": "hybrid",
        "top_k": 5
    }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Validate mode
    allowed_modes = ["hybrid", "bm25", "dense"]
    if request.mode not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Choose from: {allowed_modes}"
        )

    # Call the full RAG pipeline
    result = query_rag(
        question=request.question,
        mode=request.mode,
        top_k=request.top_k
    )

    # Convert citation dicts to Citation objects
    citations = [
        Citation(
            file_name=c["file_name"],
            drive_url=c["drive_url"],
            snippet=c["snippet"]
        )
        for c in result.get("citations", [])
    ]

    return QueryResponse(
        answer=result["answer"],
        citations=citations,
        blocked=result.get("blocked", False),
        mode=request.mode,
        question=request.question
    )


# ─────────────────────────────────────────
# POST /ingest
# ─────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Triggers the full ingestion pipeline:
    1. Downloads PDFs from Google Drive
    2. Extracts text and chunks them
    3. Indexes chunks into Elasticsearch

    Optionally accepts a folder_id to override the default.
    """
    try:
        # Override folder ID if provided
        if request.folder_id:
            os.environ["OVERRIDE_FOLDER_ID"] = request.folder_id

        chunks = run_ingestion()

        return IngestResponse(
            status="success",
            chunks_count=len(chunks),
            message=f"Successfully ingested and indexed {len(chunks)} chunks"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )