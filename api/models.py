from pydantic import BaseModel
from typing import List, Optional


# --- Request models (what the API receives) ---

class QueryRequest(BaseModel):
    """
    What the user sends when asking a question.
    mode is optional — defaults to hybrid if not provided.
    """
    question: str
    mode: Optional[str] = "hybrid"
    top_k: Optional[int] = 5


class IngestRequest(BaseModel):
    """
    What the user sends when triggering ingestion.
    folder_id is optional — uses the default from ingest.py if not provided.
    """
    folder_id: Optional[str] = None


# --- Response models (what the API returns) ---

class Citation(BaseModel):
    """
    A single citation object — one source chunk used in the answer.
    """
    file_name: str
    drive_url: str
    snippet:   str


class QueryResponse(BaseModel):
    """
    Full response returned by POST /query
    """
    answer:    str
    citations: List[Citation]
    blocked:   bool
    mode:      str
    question:  str


class IngestResponse(BaseModel):
    """
    Response returned by POST /ingest
    """
    status:       str
    chunks_count: int
    message:      str


class HealthResponse(BaseModel):
    """
    Response returned by GET /healthz
    """
    status:        str
    elasticsearch: str
    ollama:        str