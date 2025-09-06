from langchain_core.messages import AIMessage, BaseMessage
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
)

from medrag.config import settings


def _final_answer_from_messages(msgs: list[BaseMessage]) -> str:
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            return m.content


helpfulness_evaluator = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    model=settings.EVAL_MODEL,
)


async def helpfulness(inputs: dict, outputs: dict, reference_outputs: dict):
    eval_result = helpfulness_evaluator(
        inputs=inputs["messages"][0]["content"],
        outputs=_final_answer_from_messages(outputs["messages"]),
    )

    return eval_result


correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model=settings.EVAL_MODEL,
)


async def correctness(inputs: dict, outputs: dict, reference_outputs: dict):
    eval_result = correctness_evaluator(
        inputs=inputs["messages"][0]["content"],
        outputs=_final_answer_from_messages(outputs["messages"]),
        reference_outputs=reference_outputs["answer"],
    )

    return eval_result
