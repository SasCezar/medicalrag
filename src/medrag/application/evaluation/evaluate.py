from dotenv import load_dotenv

load_dotenv()

import asyncio

from langsmith import aevaluate

from medrag.application.chat.workflow.graph import graph
from medrag.application.evaluation.metrics import (
    correctness,
    hallucination,
    helpfulness,
    relevance,
    to_flat,
)


async def run_evaluation():
    target = graph | to_flat
    experiment_results = await aevaluate(
        target,
        data="medrag-singleturn-v1",
        evaluators=[
            correctness,
            relevance,
            hallucination,
            helpfulness,
        ],
    )
    return experiment_results


if __name__ == "__main__":
    res = asyncio.run(run_evaluation())
