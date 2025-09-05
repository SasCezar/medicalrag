from dotenv import load_dotenv
from langchain.globals import set_verbose
from langchain_openai import OpenAI

from medrag.application.rag.retrievers import get_retriever
from medrag.config import settings

set_verbose(True)

load_dotenv()

if __name__ == "__main__":
    llm = OpenAI(temperature=0)
    retriever = get_retriever(settings.DENSE_EMBEDDING_MODEL)
    res = retriever.query_constructor.invoke("Summarize the record of the patient named: Lucio648 Bruen238")
    print(res)

    translator = retriever.structured_query_translator  # already set if you passed one
    translated = translator.visit_structured_query(res)
    print("Translated for backend:", translated)
