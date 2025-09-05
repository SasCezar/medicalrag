from langchain_huggingface import HuggingFaceEmbeddings

EmbeddingsModel = HuggingFaceEmbeddings


def get_embedding_model(
    model_id: str,
) -> EmbeddingsModel:
    return get_huggingface_embedding_model(model_id)


def get_huggingface_embedding_model(model_id: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": False},
    )
