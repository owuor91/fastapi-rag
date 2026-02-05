from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    OPEN_AI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-5-mini"
    LLM_PROVIDER: str = "openai"  # Options: 'openai', 'anthropic'

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    COLLECTION_NAME: str = "documents"
    VECTOR_DB_PATH: str = "./chroma_db"

    TOP_K_RESULTS: int = 3

    APP_NAME: str = "FastAPI RAG System"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
