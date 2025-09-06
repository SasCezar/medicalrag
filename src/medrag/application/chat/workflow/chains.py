from pydantic import BaseModel, Field

from medrag.application.chat.workflow.utils import get_chat_model
from medrag.domain.prompts import (
    conversation_summary_prompt,
    document_grader_prompt,
    question_rewrite_prompt,
    response_generate_prompt,
    retrieval_router_prompt,
)


def get_conversation_summary_chain():
    model = get_chat_model()
    return conversation_summary_prompt | model


def get_document_grader_chain():
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    model = get_chat_model().with_structured_output(GradeDocuments)
    return document_grader_prompt | model


def get_response_generate_chain():
    model = get_chat_model()
    return response_generate_prompt | model


def get_question_rewrite_chain():
    model = get_chat_model()
    return question_rewrite_prompt | model


def get_retrieval_router_chain():
    class RetrievalDecision(BaseModel):
        """Binary decision on whether retrieval is needed."""

        need_retrieval: str = Field(
            description="Answer strictly 'yes' if external patient medical documents are needed; otherwise 'no'."
        )

    model = get_chat_model().with_structured_output(RetrievalDecision)

    return retrieval_router_prompt | model
