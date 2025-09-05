from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from medrag.config import settings


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    llm = init_chat_model(**settings.CHAT_MODEL_KWARGS)
    return llm
