from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CONCISENESS_PROMPT,
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


def flatten_state(outputs: dict[str, Any]) -> dict[str, Any]:
    prediction = _final_answer_from_messages(outputs.get("messages", []))
    context = outputs.get("context") or _concat_docs(outputs.get("documents", []))
    retrieved = _concat_docs(outputs.get("documents", []))
    return {"prediction": prediction, "context": context, "retrieved": retrieved}


to_flat = RunnableLambda(flatten_state)

conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    model=settings.EVAL_MODEL,
)


def conciseness(
    inputs: dict,
    outputs: dict,
    # Unused for this evaluator
    reference_outputs: dict,
):
    eval_result = conciseness_evaluator(
        inputs=inputs,
        outputs=outputs,
    )
    return eval_result
