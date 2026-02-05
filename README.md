# FastAPI RAG System

A **Retrieval-Augmented Generation (RAG)** API built with FastAPI. Upload documents (PDF, TXT, DOCX), embed them with sentence transformers, store them in ChromaDB, and query them using an LLM (OpenAI or Anthropic) for context-aware answers.

## Features

- **Document ingestion**: Upload PDF, plain text (.txt, .md), and Word (.docx, .doc) files.
- **Chunking & embeddings**: Automatic text chunking and embedding via sentence-transformers (e.g. `all-MiniLM-L6-v2`).
- **Vector store**: ChromaDB for persistent vector storage and similarity search (cosine similarity).
- **LLM-backed answers**: Generate answers from retrieved chunks using OpenAI or Anthropic.
- **Source attribution**: Each answer includes the retrieved chunks and similarity scores.
- **CORS enabled**: Ready for frontend or cross-origin API calls.

## Tech Stack

| Component        | Technology                          |
|-----------------|-------------------------------------|
| API             | FastAPI, Uvicorn                    |
| Embeddings      | sentence-transformers               |
| Vector DB       | ChromaDB                            |
| Document loaders| LangChain (PyPDF, Text, Docx2txt)   |
| LLM             | OpenAI and/or Anthropic             |
| Config          | pydantic-settings, `.env`           |

## Project Structure

```
fastapi-rag/
├── app/
│   ├── __init__.py
│   ├── config.py          # Settings and env config
│   ├── main.py            # FastAPI app, routes, lifespan
│   ├── models.py          # Pydantic request/response models
│   ├── routers/           # (Optional) route modules
│   └── services/
│       ├── document_processor.py  # Load & chunk documents
│       ├── embeddings.py          # Sentence transformer embeddings
│       ├── llm.py                 # OpenAI / Anthropic LLM
│       └── vector_store.py        # ChromaDB client
├── requirements.txt
├── .env                    # Your secrets (not committed)
└── README.md
```

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **API keys** (at least one):
  - **OpenAI**: [API key](https://platform.openai.com/api-keys) for models like `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`
  - **Anthropic**: [API key](https://console.anthropic.com/) for Claude models

## Installation

1. **Clone and enter the project**

   ```bash
   git clone https://github.com/owuor91/fastapi-rag.git
   cd fastapi-rag
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Note: The first run will download the sentence-transformers embedding model (e.g. ~80MB for `all-MiniLM-L6-v2`).

## Configuration

Create a `.env` file in the project root (see `.env.example` below if you add one). All settings can be overridden by environment variables.

| Variable            | Description                              | Default |
|---------------------|------------------------------------------|---------|
| `OPEN_AI_API_KEY`   | OpenAI API key (required if using OpenAI)| `""`    |
| `ANTHROPIC_API_KEY` | Anthropic API key (required for Anthropic)| `""`  |
| `LLM_PROVIDER`      | `openai` or `anthropic`                  | `openai`|
| `LLM_MODEL`         | Model name (e.g. `gpt-4o-mini`, `claude-3-5-sonnet-20241022`) | `gpt-3.5-turbo` |
| `EMBEDDING_MODEL`   | Sentence-transformers model name        | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE`        | Characters per chunk                     | `1000`  |
| `CHUNK_OVERLAP`     | Overlap between chunks                   | `200`   |
| `TOP_K_RESULTS`     | Number of chunks to retrieve per query   | `3`     |
| `VECTOR_DB_PATH`    | ChromaDB persistence directory          | `./chroma_db` |
| `COLLECTION_NAME`   | ChromaDB collection name                 | `documents` |

**Example `.env`**

```env
OPEN_AI_API_KEY=sk-your-openai-key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Optional: use Anthropic instead
# ANTHROPIC_API_KEY=your-anthropic-key
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-5-sonnet-20241022
```

## Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **API**: http://localhost:8000  
- **Interactive docs**: http://localhost:8000/docs  
- **ReDoc**: http://localhost:8000/redoc  

On startup, the app loads the embedding model and connects to ChromaDB; the first request may be slower while the model warms up.

## API Reference

### `GET /`

Root endpoint with API info and list of endpoints.

**Response**

```json
{
  "message": "RAG System API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "upload": "/documents/upload",
    "query": "/query"
  }
}
```

---

### `GET /health`

Health check and basic vector DB stats.

**Response**

```json
{
  "status": "healthy",
  "vector_db_status": "connected",
  "total_documents": 42
}
```

---

### `POST /documents/upload`

Upload a document for ingestion. The file is chunked, embedded, and stored in ChromaDB.

**Request**

- **Content-Type**: `multipart/form-data`
- **Body**: `file` — PDF, `.txt`, `.md`, `.docx`, or `.doc`

**Example (curl)**

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/your/document.pdf"
```

**Response**

```json
{
  "file_name": "document.pdf",
  "chunks_created": 15,
  "document_id": "document.pdf",
  "message": "Document uploaded and processed successfully."
}
```

**Errors**

- `400`: Unsupported file type
- `500`: File save failure or processing error (e.g. embedding/chunking)

---

### `POST /query`

Ask a question and get an answer based on the retrieved document chunks.

**Request**

- **Content-Type**: `application/json`
- **Body**:
  - `question` (string, required): The question to answer.
  - `top_k` (integer, optional): Number of chunks to retrieve (default from config, e.g. `3`).

**Example (curl)**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics in chapter 7?", "top_k": 5}'
```

**Response**

```json
{
  "answer": "Chapter 7 covers transactions, concurrency control, and isolation levels...",
  "source_chunks": [
    {
      "content": "In this chapter, we will examine many examples of things that can go wrong...",
      "source": "Designing Data-Intensive Applications.pdf",
      "chunk_id": 706,
      "similarity_score": 0.6998342275619507
    }
  ]
}
```

If no documents are in the store, `answer` will indicate that and `source_chunks` will be `[]`.

---

### `DELETE /documents/clear`

Delete the current ChromaDB collection and reinitialize it (all ingested documents are removed).

**Response**

```json
{
  "message": "All documents cleared successfully."
}
```

## Usage Flow

1. **Upload documents**  
   `POST /documents/upload` with one or more PDF/TXT/DOCX files.

2. **Query**  
   `POST /query` with a JSON body `{"question": "Your question?", "top_k": 3}`.  
   The API embeds the question, retrieves the top-k chunks, and asks the LLM to answer using only that context.

3. **Optional: clear and re-ingest**  
   `DELETE /documents/clear` to wipe the vector store, then upload again.

## Development

- **Data directories**: Uploaded files are stored under `data/documents/`; the vector DB is stored at `VECTOR_DB_PATH` (default `./chroma_db/`). Both are ignored or local; add them to `.gitignore` if needed (e.g. `data/`, `chroma_db/`).
- **CORS**: The app allows all origins (`allow_origins=["*"]`). Tighten this in production.
- **Secrets**: Never commit `.env`. Use environment variables or a secrets manager in production.

## License

MIT (or your chosen license).
