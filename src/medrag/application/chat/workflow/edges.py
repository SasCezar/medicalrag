from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END
from loguru import logger

from medrag.application.chat.workflow.state import ConversationState

from .chains import get_retrieval_router_chain


async def should_summarize(state: ConversationState) -> Literal["summarize_conversation", END]:
    last = state.get("messages")[-1].content
    turn = state.get("turn_count", 0)
    logger.debug(f"Router: Summarize - turn={turn} last_len={len(last)}")
    if (turn > 0 and turn % 3 == 0) or len(last) > 500:
        return "summarize_conversation"
    return END


async def decide_retrieval(state: ConversationState) -> Literal["rewrite_question", "generate"]:
    logger.info("Router: decide_retrieval")

    q = state.get("question") or next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
    )

    try:
        chain = get_retrieval_router_chain()
        result = await chain.ainvoke(
            {
                "question": q,
                "summary": state.get("summary", ""),
                "messages": state.get("messages", []),
            }
        )
        print(result)
        retireve = result.need_retrieval.strip().lower().startswith("y")
    except Exception as e:
        logger.exception(f"Router failed, defaulting to retrieval: {e}")
        retireve = True

    if retireve:
        return "rewrite_question"

    return "generate"
