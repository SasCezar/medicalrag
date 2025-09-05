from typing import Literal

from langchain_core.messages import RemoveMessage
from loguru import logger

from medrag.application.chat.workflow.state import ConversationState
from medrag.application.rag.retrievers import get_retriever
from medrag.config import settings

from .chains import get_conversation_summary_chain, get_document_grader_chain, get_response_generate_chain

retriever = get_retriever(embedding_model_id=settings.DENSE_EMBEDDING_MODEL)


async def should_summarize(state: ConversationState) -> Literal["do_summary", "skip_summary"]:
    last = state.get("agent_output", "") or ""
    if state.get("turn_count", 0) % 3 == 0 or len(last) > 500:
        return "do_summary"
    return "skip_summary"


async def summarize_conversation(state: ConversationState) -> ConversationState:
    summary = state.get("summary", "")
    summary_chain = get_conversation_summary_chain()

    response = await summary_chain.ainvoke(
        {
            "messages": state["messages"],
            "summary": summary,
        }
    )

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-3]]

    return {"summary": response.content, "messages": delete_messages}


async def retrieve(state: ConversationState) -> ConversationState:
    query = state["query"]
    documents = retriever.invoke(query)
    return {"documents": documents}


async def grade_docs(state: ConversationState) -> ConversationState:
    query = state["query"]
    documents = state["documents"]
    filtered_docs = []
    grade_chain = get_document_grader_chain()
    for d in documents:
        result = grade_chain.invoke({"document": d.page_content, "question": query})
        if result.binary_score == "yes":
            logger.info("Document deemed relevant.")
            filtered_docs.append(d)
        else:
            logger.info("Document deemed not relevant.")
    return {"documents": filtered_docs, "query": query, "generation": None}


async def generate(state: ConversationState) -> ConversationState:
    logger.info("Generating answer.")
    query = state["query"]
    documents = [d.page_content for d in state["documents"]]
    documents = [str(state["documents"][0].metadata)] + documents
    documents = "\n".join(documents)
    response_generation = get_response_generate_chain()
    generation = response_generation.invoke({"context": documents, "question": query})
    return {"documents": documents, "query": query, "generation": generation}
