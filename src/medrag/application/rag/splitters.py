from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from medrag.config import settings

Splitter = RecursiveCharacterTextSplitter


def get_splitter(chunk_size: int, chunk_overlap: int) -> Splitter:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=settings.ENCODING_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", "."],
    )
    return text_splitter
