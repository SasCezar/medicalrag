from typing import Literal

from langgraph.graph import END
from loguru import logger

from .state import ConversationState


def should_summarize(state: ConversationState) -> Literal["summarize_conversation", END]:
    last = state.get("messages")[-1].content
    turn = state.get("turn_count", 0)
    logger.debug(f"Router: Summarize - turn={turn} last_len={len(last)}")
    if (turn > 0 and turn % 3 == 0) or len(last) > 500:
        return "summarize_conversation"
    return END
