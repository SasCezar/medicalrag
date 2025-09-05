from typing import Literal

from langgraph.graph import END

from medrag.config import settings

from .state import ConversationState


def should_summarize_conversation(
    state: ConversationState,
) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END
