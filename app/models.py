from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    file_name: str
    chunks_created: int
    document_id: str
    message: str = "Document uploaded and processed successfully."


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="The question to ask the RAG system."
    )
    top_k: Optional[int] = Field(
        3, description="Number of chunks to retrieve."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the capital of France?",
                "top_k": 3,
            }
        }


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_id: int
    similarity_score: float


class HealthResponse(BaseModel):
    status: str
    vector_db_status: str
    total_documents: int
