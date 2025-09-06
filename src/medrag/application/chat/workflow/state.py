from typing import List, Optional

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class ConversationState(MessagesState):
    summary: Optional[str]
    question: Optional[str]
    documents: List[Document]
    turn_count: int
