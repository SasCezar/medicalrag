from langchain_core.prompts import ChatPromptTemplate

conversation_summary_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You summarize a chat. Produce a short description of the conversation so far "
            "that still captures all relevant information shared.",
        ),
        ("human", "Conversation so far:\n{messages}\n\nWrite a concise summary."),
    ]
)

document_grader_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a grader assessing the relevance of a retrieved document to a user question.\n"
            "If the document contains medical or personal information then it is relevant.\n"
            "The goal is to filter out erroneous retrievals.\n"
            "Give a binary score 'yes' or 'no' to indicate whether the document is relevant.",
        ),
        ("human", "Retrieved document:\n{document}\n\nUser question: {question}"),
    ]
)

response_generate_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. Use the provided context to answer. "
            "If you don't know, say you don't know. Keep the answer to at most three sentences.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ]
)

question_rewrite_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You summarize the user chat into a single question that can be used to collect future "
            "information from a database. Preserve personal information of patients. "
            "If multiple patients are present, keep only the last one mentioned.",
        ),
        (
            "human",
            "Chat so far:\n{messages}\n\n"
            "Summary (optional): {summary}\n\n"
            "Rewrite the user requests as a standalone question, adding any important details from past messages.",
        ),
    ]
)

retrieval_router_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a routing assistant that decides whether to consult external medical records. "
            "Answer 'yes' if the question needs domain-specific facts, citations, or patient-specific "
            "information from this patient's record. Answer 'no' for chit-chat, general knowledge, or "
            "formatting/rewrite tasks where no patient info is needed.",
        ),
        ("human", "Question: {question}\n\nConversation summary (optional): {summary}\n\nRecent messages: {messages}"),
    ]
)
