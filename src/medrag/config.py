from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", override=True)

    OPENAI_API_KEY: str

    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

    DENSE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_SIZE: int = 384

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "medical_records"

    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 64

    CHAT_MODEL: str = "openai:gpt-5-mini"
    CHAT_MODEL_KWARGS: dict = {"model": CHAT_MODEL, "temperature": 0, "seed": 42}
    ENCODING_NAME: str = "o200k_base"

    EVAL_MODEL: str = "openai:gpt-5-mini"


settings = Settings()
