from pathlib import Path

import numpy as np
import typer
from langchain_core.documents import Document
from loguru import logger

from medrag.application.data.extract import extract_documents
from medrag.application.rag.embeddings import get_embedding_model
from medrag.application.rag.splitters import get_splitter
from medrag.config import settings
from medrag.infrastructure.qdrant import QdrantVectorDB

app = typer.Typer(add_completion=True)


@app.command()
def main(dir: Path = Path("/home/sasce/PycharmProjects/MedicalRAG/data/raw/text")):
    reports: list[Document] = extract_documents(dir)
    lens = [len(r.page_content) for r in reports]
    logger.info(
        f"Document lengths: min={min(lens)}, max={max(lens)}, avg={sum(lens) / len(lens):.1f}, std={np.std(lens):.1f}"
    )
    logger.info(f"Extracted {len(reports)} documents from {dir}")
    splitter = get_splitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    docs = splitter.split_documents(reports)
    logger.info(f"Split into {len(docs)} chunks using {settings.CHUNK_SIZE=} and {settings.CHUNK_OVERLAP=}")
    embeddings = get_embedding_model(settings.DENSE_EMBEDDING_MODEL)
    with QdrantVectorDB(
        url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION_NAME, embedding_model=embeddings
    ) as vdb:
        vdb.add_documents(docs)
        logger.info(f"Ingested {len(docs)} documents into Qdrant at {settings.QDRANT_URL}")


if __name__ == "__main__":
    app()
