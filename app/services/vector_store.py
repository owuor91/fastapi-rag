import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        self.collection_name = collection_name

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> List[str]:
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadata, documents=texts
        )
        return ids

    def search(
        self, query_embedding: List[float], top_k: int = 3
    ) -> Dict[str, Any]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def delete_collection(self):
        self.client.delete_collection(name=self.collection_name)

    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
        }
