from langchain_core.documents import Document
from langgraph.graph import MessagesState


class ConversationState(MessagesState):
    query: str
    summary: str | None = None
    documents: list[Document] = []
    generation: str
