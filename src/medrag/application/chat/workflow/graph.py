from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    generate,
    grade_docs,
    retrieve,
    rewrite_question,
    should_summarize,
    summarize_conversation,
)
from .state import ConversationState


@lru_cache(maxsize=1)
def create_workflow_graph():
    builder = StateGraph(ConversationState)
    builder.set_entry_point("rewrite_question")

    builder.add_node("rewrite_question", rewrite_question)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade_docs)
    builder.add_node("generate", generate)
    builder.add_node("summarize_conversation_node", summarize_conversation)

    builder.add_edge(START, "rewrite_question")
    builder.add_edge("rewrite_question", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_edge("grade", "generate")
    builder.add_edge

    builder.add_conditional_edges(
        "generate",
        should_summarize,
        {"do_summary": "summarize_conversation_node", "skip_summary": END},
    )
    builder.add_edge("summarize_conversation_node", END)
    return builder


checkpointer = MemorySaver()
graph = create_workflow_graph().compile(checkpointer=checkpointer)
