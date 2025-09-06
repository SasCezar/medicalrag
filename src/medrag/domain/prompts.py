from langchain.prompts.chat import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

conversation_summary_prompt = ChatPromptTemplate(
    [
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(
            '"""Create a summary of the conversation. '
            "The summary must be a short description of the conversation so far, "
            'but that also captures all the relevant information shared."""'
        ),
    ]
)


document_grader_prompt = ChatPromptTemplate(
    [
        SystemMessage(
            "You are a grader assessing the relevance of a retrieved document to a user question.\n"
            "If the document contains medical or personal information then it is relevant.\n"
            "The goal is to filter out erroneous retrievals.\n"
            "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="Retrieved document: \n\n{document}\n\nUser question: {question}",
                input_variables=["document", "question"],
            ),
        ),
    ]
)


response_generate_prompt = ChatPromptTemplate(
    [
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise.\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Answer:"
                ),
                input_variables=["question", "context"],
            )
        )
    ]
)

question_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You rewrite the latest user turn into a standalone question. "
            "Keep it concise and self-contained, preserve any identifiable information or constraints. "
            "Include all relevant information for a query."
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=(
                    "Chat so far:\n\n{messages}\n\n"
                    "Summary (optional): {summary}\n\n"
                    "Rewrite the latest user request as a standalone question, "
                    "adding any information from past messages that might be important."
                ),
                input_variables=["messages", "summary"],
            )
        ),
    ]
)
