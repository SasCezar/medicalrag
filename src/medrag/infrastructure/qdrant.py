from uuid import uuid4

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)

from medrag.application.rag.retrievers import get_vectorstore


class QdrantVectorDB:
    def __init__(self, url: str, collection: str, embedding_model: HuggingFaceEmbeddings):
        self.url = url
        self.collection = collection
        self.embedding_model: HuggingFaceEmbeddings = embedding_model

    def create(self, embedding_size: int):
        client = QdrantClient(location=self.url)

        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            )

    def ingest_documents(self, documents: list):
        with self as vs:
            ids = [uuid4() for _ in documents]
            vs.add_documents(documents, collection_name=self.collection, ids=ids)

    def __enter__(self) -> QdrantVectorStore:
        dim = len(self.embedding_model.embed_query("test"))
        self.create(embedding_size=dim)
        self.client = QdrantClient(location=self.url)
        self.vs = get_vectorstore(self.embedding_model)
        return self.vs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
