from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith.client import Client

DATASET_NAME = "medrag-singleturn-v1"

_examples = [
    # Abel832 Jacobi462
    {
        "inputs": {"messages": [HumanMessage("For patient Abel832 Jacobi462, list medications.")]},
        "outputs": {"answer": "amLODIPine 2.5 MG Oral Tablet; ferrous sulfate 325 MG Oral Tablet."},
    },
    {
        "inputs": {"messages": [HumanMessage("What allergies are recorded for patient Abel832 Jacobi462?")]},
        "outputs": {"answer": "Eggs (edible) (substance); Aspirin; Allergic disposition (finding)."},
    },
    {
        "inputs": {"messages": [HumanMessage("Which care plans are active for patient Abel832 Jacobi462?")]},
        "outputs": {
            "answer": "Lifestyle education regarding hypertension; Diabetes self management plan; Self-care interventions."
        },
    },
    # Brittny484 Koepp521
    {
        "inputs": {"messages": [HumanMessage("For patient Brittny484 Koepp521, list medications.")]},
        "outputs": {
            "answer": "amLODIPine 2.5 MG Oral Tablet; lisinopril 10 MG Oral Tablet; Hydrochlorothiazide 25 MG Oral Tablet."
        },
    },
    {
        "inputs": {"messages": [HumanMessage("Does patient Brittny484 Koepp521 have any recorded allergies?")]},
        "outputs": {"answer": "No Known Allergies."},
    },
    {
        "inputs": {"messages": [HumanMessage("What is the tobacco smoking status for patient Brittny484 Koepp521?")]},
        "outputs": {"answer": "Never smoked tobacco (finding)."},
    },
    # Carlotta746 Emely698 Feeney44
    {
        "inputs": {"messages": [HumanMessage("For patient Carlotta746 Emely698 Feeney44, list medications.")]},
        "outputs": {"answer": "Furosemide 40 MG Oral Tablet."},
    },
    {
        "inputs": {
            "messages": [HumanMessage("Which care plans are active for patient Carlotta746 Emely698 Feeney44?")]
        },
        "outputs": {"answer": "Heart failure self management plan; Diabetes self management plan."},
    },
    {
        "inputs": {"messages": [HumanMessage("What is the cause of death for patient Carlotta746 Emely698 Feeney44?")]},
        "outputs": {"answer": "Chronic congestive heart failure (disorder)."},
    },
]

load_dotenv()


def create_dataset():
    client = Client()
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Questions and answers about patients.",
    )

    client.create_examples(dataset_id=dataset.id, examples=_examples)


if __name__ == "__main__":
    create_dataset()
