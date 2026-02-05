from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
from datetime import datetime, timezone

from app.config import get_settings, Settings

from app.models import (
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm import LLMService
from app.services.document_processor import DocumentProcessor
from contextlib import asynccontextmanager

embedding_service = None
vector_store = None
llm_service = None
document_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service, vector_store, llm_service, document_processor
    settings = get_settings()

    print("Starting up and initializing services...")

    embedding_service = EmbeddingService(
        model_name=settings.EMBEDDING_MODEL
    )

    vector_store = VectorStore(
        persist_directory=settings.VECTOR_DB_PATH,
        collection_name=settings.COLLECTION_NAME,
    )

    document_processor = DocumentProcessor(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )

    api_key = (
        settings.OPEN_AI_API_KEY
        if settings.LLM_PROVIDER.lower() == "openai"
        else settings.ANTHROPIC_API_KEY
    )

    llm_service = LLMService(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        api_key=api_key,
    )

    print("All services initialized successfully.")
    yield
    print("Shutting down...")


app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def read_root():
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/documents/upload",
            "query": "/query",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    stats = vector_store.get_collection_stats()
    return HealthResponse(
        status="healthy",
        vector_db_status="connected",
        total_documents=stats["total_documents"],
    )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Allowed types are: {', '.join(allowed_extensions)}",
        )

    os.makedirs("data/documents", exist_ok=True)

    file_path = f"data/documents/{file.filename}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    try:
        print(f"Processing document: {file.filename}")
        chunks = document_processor.process_document(file_path)
        print(f"Document processed into {len(chunks)} chunks.")

        print("Generating embeddings for chunks...")
        embeddings = embedding_service.embed_batch(chunks)
        print(f"Created {len(embeddings)} embeddings.")

        metadata = [
            {
                "source": file.filename,
                "chunk_id": idx,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            for idx in range(len(chunks))
        ]

        print("Adding documents to vector db...")
        doc_ids = vector_store.add_documents(
            texts=chunks, embeddings=embeddings, metadata=metadata
        )
        print(f"Added {len(doc_ids)} documents to vector store.")

        return DocumentUploadResponse(
            file_name=file.filename,
            chunks_created=len(chunks),
            document_id=file.filename,
            message="Document uploaded and processed successfully.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest,
    settings: Settings = Depends(get_settings),
):
    try:
        print(f"Processing query: {query_request.question}")
        query_embedding = embedding_service.embed_text(query_request.question)

        top_k = query_request.top_k or settings.TOP_K_RESULTS

        print("Searching for {top_k} relevant chunks...")
        search_results = vector_store.search(
            query_embedding=query_embedding, top_k=top_k
        )

        retrieved_chunks = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]
        distances = search_results["distances"][0]

        if not retrieved_chunks:
            return QueryResponse(
                answer="No relevant documents found. Please upload documents first.",
                source_chunks=[],
            )

        print("Generating answer from LLM...")

        answer = llm_service.generate_answer(
            question=query_request.question,
            context_chunks=retrieved_chunks,
        )

        sources = [
            SourceChunk(
                content=chunk,
                source=metadata["source"],
                chunk_id=metadata["chunk_id"],
                similarity_score=1 - distance,
            )
            for chunk, metadata, distance in zip(
                retrieved_chunks, metadatas, distances
            )
        ]

        return QueryResponse(answer=answer, source_chunks=sources)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process query: {str(e)}"
        )


@app.delete("/documents/clear", response_model=dict)
async def clear_documents(settings: Settings = Depends(get_settings)):
    global vector_store
    try:
        vector_store.delete_collection()
        vector_store = VectorStore(
            persist_directory=settings.VECTOR_DB_PATH,
            collection_name=settings.COLLECTION_NAME,
        )
        return {"message": "All documents cleared successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear documents: {str(e)}"
        )
