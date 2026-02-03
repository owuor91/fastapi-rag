from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingService:
    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )
        print(
            f"Model loaded with embedding dimension: {self.embedding_dimension}"
        )

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension
