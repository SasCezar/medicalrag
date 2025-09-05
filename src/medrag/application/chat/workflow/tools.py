from langchain.tools.retriever import create_retriever_tool

from medrag.application.rag.retrievers import get_retriever
from medrag.config import settings

retriever = get_retriever(embedding_model_id=settings.DENSE_EMBEDDING_MODEL)

retriever_tool = create_retriever_tool(
    retriever,
    "patient_record_retireval",
    "Uses an LLM to query a database for the patients record. Provide the full user context not a query.",
)

tools = [retriever_tool]
