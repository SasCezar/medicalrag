from langsmith import aevaluate

from medrag.application.chat.workflow.graph import create_workflow_graph
from medrag.application.evaluation.metrics import conciseness, to_flat

graph = create_workflow_graph().compile()


async def run_evaluation():
    target = graph | to_flat
    experiment_results = await aevaluate(
        target,
        data="medrag-singleturn-v1",
        evaluators=[
            conciseness,
        ],
    )
    return experiment_results
