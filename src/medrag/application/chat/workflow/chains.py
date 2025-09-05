from langchain import hub
from langchain.prompts.chat import HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from medrag.application.chat.workflow.utils import get_chat_model
from medrag.domain.prompts import SUMMARY_PROMPT


def get_conversation_summary_chain():
    model = get_chat_model()

    summary_message = SUMMARY_PROMPT

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("human", summary_message),
        ],
        template_format="jinja2",
    )

    return prompt | model


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


def get_document_grader_chain():
    model = get_chat_model()
    model = model.with_structured_output(GradeDocuments)
    prompt = ChatPromptTemplate(
        [
            SystemMessage(
                "You are a grader assessing the relevance of a retrieved document to a user question.\n"
                "If the document contains kmedical or personal information then is relevant.\n"
                "The goal is to filter out erroneous retrievals.\n"
                "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template="Retrieved document: \n\n {document} \n\n User question: {question}",
                    input_variables=["document", "question"],
                ),
            ),
        ]
    )
    return prompt | model


def get_response_generate_chain():
    rag_prompt = hub.pull("rlm/rag-prompt")
    model = get_chat_model()
    return rag_prompt | model | StrOutputParser()
