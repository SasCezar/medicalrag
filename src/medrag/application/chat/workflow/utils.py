from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from medrag.config import settings


def get_chat_model() -> BaseChatModel:
    llm = init_chat_model(**settings.CHAT_MODEL_KWARGS)
    return llm
