from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .edges import decide_retrieval, should_summarize
from .nodes import (
    generate,
    grade_docs,
    no_op,
    retrieve,
    rewrite_question,
    summarize_conversation,
)
from .state import ConversationState


@lru_cache(maxsize=1)
def create_workflow_graph():
    builder = StateGraph(ConversationState)
    builder.set_entry_point("router")

    builder.add_node("router", no_op)
    builder.add_node("rewrite_question", rewrite_question)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade_docs)
    builder.add_node("generate", generate)
    builder.add_node("summarize_conversation", summarize_conversation)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", decide_retrieval)
    builder.add_edge("rewrite_question", "retrieve")
    # builder.add_conditional_edges("rewrite_question", decide_retrieval)
    builder.add_edge("retrieve", "grade")
    builder.add_edge("grade", "generate")
    builder.add_conditional_edges("generate", should_summarize)
    builder.add_edge("summarize_conversation", END)
    return builder
