from __future__ import annotations

import asyncio
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from loguru import logger

from medrag.application.chat.workflow.state import ConversationState
from medrag.application.rag.retrievers import get_retriever
from medrag.config import settings

from .chains import (
    get_conversation_summary_chain,
    get_document_grader_chain,
    get_question_rewrite_chain,
    get_response_generate_chain,
)

retriever = get_retriever(embedding_model_id=settings.DENSE_EMBEDDING_MODEL)


def last_human_text(messages: Sequence[BaseMessage]) -> str:
    """Return last HumanMessage.content or empty string."""
    return next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )


def current_question(state: ConversationState) -> str:
    """Prefer explicit state['question']; fallback to the last human turn."""
    return (state.get("question") or last_human_text(state["messages"]) or "").strip()


def summarize_context(docs) -> str:
    """Build the context string from retrieved docs."""
    if not docs:
        return ""
    patient_info = str(docs[0].metadata)
    reports = [d.page_content for d in docs]
    return "\n\n".join([patient_info, *reports])


async def summarize_conversation(state: ConversationState) -> ConversationState:
    logger.info("Node: summarize_conversation")
    chain = get_conversation_summary_chain()
    resp = await chain.ainvoke(
        {
            "messages": state["messages"],
            "summary": state.get("summary", ""),
        }
    )
    # keep the last few messages; drop earlier ones
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-3]]
    return {"summary": getattr(resp, "content", ""), "messages": delete_messages}


async def rewrite_question(state: ConversationState) -> ConversationState:
    logger.info("Node: rewrite_question")
    chain = get_question_rewrite_chain()
    try:
        result = await chain.ainvoke(
            {
                "messages": state["messages"],
                "summary": state.get("summary", ""),
            }
        )
        question = (getattr(result, "content", None) or str(result)).strip()
    except Exception as e:
        logger.exception(f"Rewrite failed, fallback to last user: {e}")
        question = last_human_text(state["messages"])
    return {"question": question, "messages": [HumanMessage(question)]}


async def retrieve(state: ConversationState) -> ConversationState:
    logger.info("Node: retrieve")
    q = current_question(state)
    try:
        docs = await retriever.ainvoke(q)
    except Exception as e:
        logger.exception(f"Retriever failed: {e}")
        docs = []
    logger.debug(f"Retrieved {len(docs)} docs")
    return {"documents": docs}


async def grade_docs(state: ConversationState) -> ConversationState:
    logger.info("Node: grade_docs")
    q = current_question(state)
    grade = get_document_grader_chain()

    async def grade_one(doc):
        try:
            r = await grade.ainvoke({"document": doc.page_content, "question": q})
            return (getattr(r, "binary_score", "")).strip().lower() == "yes", doc
        except Exception as e:
            logger.exception(f"Grade error, dropping doc: {e}")
            return False, doc

    tasks = [grade_one(d) for d in state.get("documents", [])]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    filtered = [doc for ok, doc in results if ok]

    logger.debug(f"Kept {len(filtered)}/{len(state.get('documents', []))}")
    return {"documents": filtered}


async def generate(state: ConversationState) -> ConversationState:
    logger.info("Node: generate")

    q = current_question(state)
    context = summarize_context(state.get("documents", []))

    chain = get_response_generate_chain()
    try:
        message = await chain.ainvoke({"context": context, "question": q})
    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        message = AIMessage("Sorry, I hit an error while generating the answer.")

    return {
        "messages": [message],
        "context": context,
        "question": None,
        "summary": state.get("summary", ""),
    }


async def no_op(state: ConversationState) -> ConversationState:
    return state
