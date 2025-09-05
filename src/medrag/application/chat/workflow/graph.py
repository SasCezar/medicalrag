from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .nodes import generate, grade_docs, retrieve
from .state import ConversationState


@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(ConversationState)
    graph_builder.set_entry_point("retrieve")
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("grade", grade_docs)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "grade")
    graph_builder.add_edge("grade", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder


graph = create_workflow_graph().compile()
