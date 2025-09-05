from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
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


def should_summarize(state: ConversationState) -> Literal["do_summary", "skip_summary"]:
    last = state.get("agent_output", "")
    turn = state.get("turn_count", 0)
    logger.debug(f"[Router] turn={turn} last_len={len(last)}")
    if (turn > 0 and turn % 3 == 0) or len(last) > 500:
        return "do_summary"
    return "skip_summary"


async def summarize_conversation(state: ConversationState) -> ConversationState:
    logger.info("Node: summarize_conversation")
    chain = get_conversation_summary_chain()
    resp = await chain.ainvoke({"messages": state["messages"], "summary": state.get("summary", "")})
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-3]]
    return {"summary": resp.content, "messages": delete_messages}


async def rewrite_question(state: ConversationState) -> ConversationState:
    logger.info("Node: rewrite_question")
    chain = get_question_rewrite_chain()
    try:
        result = await chain.ainvoke({"messages": state["messages"], "summary": state.get("summary", "")})
        question = getattr(result, "content", None) or str(result)
    except Exception as e:
        logger.exception(f"Rewrite failed, fallback to last user: {e}")
        question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    human = [HumanMessage(question)]
    return {"question": question, "messages": human}


async def retrieve(state: ConversationState) -> ConversationState:
    logger.info("Node: retrieve")
    q = state["question"] or next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    try:
        docs = await retriever.ainvoke(q)
    except Exception as e:
        logger.exception(f"Retriever failed: {e}")
        docs = []
    logger.debug(f"Retrieved {len(docs)} docs")
    return {"documents": docs}


async def grade_docs(state: ConversationState) -> ConversationState:
    logger.info("Node: grade_docs")
    q = state["question"] or next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    grade = get_document_grader_chain()
    filtered = []
    for d in state["documents"]:
        try:
            r = await grade.ainvoke({"document": d.page_content, "question": q})
            ok = (getattr(r, "binary_score", "") or "").strip().lower() == "yes"
        except Exception as e:
            logger.exception(f"Grade error, dropping doc: {e}")
            ok = False
        if ok:
            filtered.append(d)
    logger.debug(f"Kept {len(filtered)}/{len(state['documents'])}")
    return {"documents": filtered}


async def generate(state: ConversationState) -> ConversationState:
    logger.info("Node: generate")
    q = state["question"] or next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    if not state["documents"]:
        context = "No directly relevant documents were found."
    else:
        parts = [str(state["documents"][0].metadata)]
        parts.extend([d.page_content for d in state["documents"]])
        context = "\n\n".join(parts)

    chain = get_response_generate_chain()
    try:
        out = await chain.ainvoke({"context": context, "question": q})
        text = out if isinstance(out, str) else getattr(out, "content", str(out))
    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        text = "Sorry, I hit an error while generating the answer."

    ai = AIMessage(content=text)
    return {
        "messages": [ai],
        "context": context,
        "question": None,
        "summary": state.get("summary", ""),
    }
