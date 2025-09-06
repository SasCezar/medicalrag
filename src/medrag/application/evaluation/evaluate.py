from langsmith import aevaluate

from medrag.application.chat.workflow.graph import create_workflow_graph
from medrag.application.evaluation.metrics import correctness, helpfulness

graph = create_workflow_graph().compile()


async def run_evaluation():
    target = graph
    experiment_results = await aevaluate(
        target,
        data="medrag-singleturn-v1",
        evaluators=[helpfulness, correctness],
    )
    return experiment_results
