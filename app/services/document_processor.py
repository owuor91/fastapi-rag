from typing import List, BinaryIO
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_document(self, file_path: str) -> List[str]:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = loader.load()
        return [doc.page_content for doc in documents]

    def chunk_text(self, text: str) -> List[str]:
        chunks = self.text_splitter.split_text(text)
        return chunks

    def process_document(self, file_path: str) -> List[str]:
        pages = self.load_document(file_path)
        full_text = "\n\n".join(pages)
        chunks = self.chunk_text(full_text)
        return chunks
