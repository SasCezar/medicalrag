from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chat_models import init_chat_model
from langchain.retrievers import SelfQueryRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from loguru import logger

from medrag.config import settings

from .embeddings import get_embedding_model

Retriever = VectorStoreRetriever


def get_retriever(
    embedding_model_id: str,
) -> Retriever:
    logger.info(f"Initializing retriever | model: {embedding_model_id}")

    embedding_model = get_embedding_model(embedding_model_id)
    vector_store = get_vectorstore(embedding_model)
    return get_search_retriever(vector_store)


def get_search_retriever(vectorstore: QdrantVectorStore) -> VectorStoreRetriever:
    metadata_info = [
        AttributeInfo(
            name="name",
            description=(
                "Full patient name (often with an alphanumeric identifier). Filterable by exact or partial match."
            ),
            type="string",
        ),
        AttributeInfo(
            name="race",
            description="Patient's race as recorded in the chart (e.g., 'White').",
            type="string",
        ),
        AttributeInfo(
            name="ethnicity",
            description="Patient's ethnicity as recorded in the chart (e.g., 'Non-Hispanic').",
            type="string",
        ),
        AttributeInfo(
            name="gender",
            description=("Patient's gender; short string normalized to uppercase (e.g., 'M', 'F')."),
            type="string",
        ),
        AttributeInfo(
            name="age",
            description="Patient age in years (integer ≥ 0). Supports numeric comparisons.",
            type="integer",
        ),
        AttributeInfo(
            name="birth_date",
            description="Patient date of birth in 'YYYY-MM-DD' format. Supports date comparisons.",
            type="date",
        ),
        AttributeInfo(
            name="marital_status",
            description="Patient's marital status (e.g., 'S', 'M', 'Divorced').",
            type="string",
        ),
        AttributeInfo(
            name="type",
            description=(
                "Section of the record from which the clinical entry was extracted. "
                "One of: 'ALLERGIES', 'MEDICATIONS', 'CONDITIONS', 'CARE PLANS', "
                "'REPORTS', 'OBSERVATIONS', 'PROCEDURES', 'IMMUNIZATIONS', "
                "'ENCOUNTERS', 'IMAGING STUDIES'."
            ),
            type="string",
        ),
    ]

    retriever = SelfQueryRetriever.from_llm(
        llm=init_chat_model(**settings.CHAT_MODEL_KWARGS),
        vectorstore=vectorstore,
        document_contents="Medical test results and medical documents",
        metadata_field_info=metadata_info,
        verbose=True,
    )

    return retriever


def get_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
):
    vectorstore = QdrantVectorStore.from_existing_collection(
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=embedding_model,
        retrieval_mode=RetrievalMode.DENSE,
    )

    return vectorstore
