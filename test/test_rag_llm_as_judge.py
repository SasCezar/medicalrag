from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT

client = Client()

conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    model="openai:o3-mini",
)

def wrapped_conciseness_evaluator(
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

experiment_results = client.evaluate(
    # This is a dummy target function, replace with your actual LLM-based system
    lambda inputs: "What color is the sky?",
    data="Sample dataset",
    evaluators=[
        wrapped_conciseness_evaluator
    ]
)
