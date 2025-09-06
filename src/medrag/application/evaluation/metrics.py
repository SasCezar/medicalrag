from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    HALLUCINATION_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

from medrag.config import settings


def _final_answer_from_messages(msgs: list[BaseMessage]) -> str:
    for m in reversed(msgs or []):
        if isinstance(m, AIMessage) and (m.content or "").strip():
            return str(m.content)
    return str((msgs or [])[-1].content) if msgs else ""


def _concat_docs(docs: list[Document]) -> str:
    if not docs:
        return ""
    return "\n\n---\n\n".join((f"[meta={getattr(d, 'metadata', None)}]\n{d.page_content}").strip() for d in docs)


def to_state(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"question": [HumanMessage(content=inputs["question"])]}


def flatten_state(outputs: dict[str, Any]) -> dict[str, Any]:
    prediction = _final_answer_from_messages(outputs.get("messages", []))
    context = outputs.get("context") or _concat_docs(outputs.get("documents", []))
    retrieved = _concat_docs(outputs.get("documents", []))
    return {"prediction": prediction, "context": context, "retrieved": retrieved}


to_flat = RunnableLambda(flatten_state)


adapt_input = RunnableLambda(to_state)

_correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model=settings.EVAL_MODEL,
)

_helpfulness = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    model=settings.EVAL_MODEL,
)

_groundedness = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    model=settings.EVAL_MODEL,
)

_retrieval_rel = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
    model=settings.EVAL_MODEL,
)

_hallucination = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    feedback_key="hallucination",
    model=settings.EVAL_MODEL,
)


def correctness(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]):
    # expected/actual naming kept explicit for clarity
    return _correctness_evaluate(
        question=inputs["question"],
        expected=reference_outputs.get("answer", ""),
        actual=outputs["prediction"],
    )


def _correctness_evaluate(question: str, expected: str, actual: str):
    res = _correctness(
        inputs={"question": question, "expected_answer": expected},
        outputs={"actual_answer": actual},
        reference_outputs={},
    )
    if isinstance(res, dict):
        res.setdefault("key", "correctness")
    return res


def helpfulness(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]):
    res = _helpfulness(
        inputs={"question": inputs["question"], "context": outputs["context"]},
        outputs={"answer": outputs["prediction"]},
        reference_outputs={},
    )
    if isinstance(res, dict):
        res.setdefault("key", "helpfulness")
    return res


def groundedness(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]):
    res = _groundedness(
        inputs={"context": outputs["context"], "retrieved_snippets": outputs["retrieved"]},
        outputs={"answer": outputs["prediction"]},
        reference_outputs={},
    )
    if isinstance(res, dict):
        res.setdefault("key", "groundedness")
    return res


def relevance(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]):
    res = _retrieval_rel(
        inputs={"question": inputs["question"], "retrieved_snippets": outputs["retrieved"]},
        outputs={},
        reference_outputs={},
    )
    if isinstance(res, dict):
        res.setdefault("key", "retrieval_relevance")
    return res


def hallucination(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]):
    res = _hallucination(
        inputs={"question": inputs["question"], "context": outputs["context"]},
        outputs={"answer": outputs["prediction"]},
        reference_outputs={},
    )
    if isinstance(res, dict):
        res.setdefault("key", "hallucination")
    return res
