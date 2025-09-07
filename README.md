# MedicalRAG

MedicalRAG is a **Retrieval-Augmented Generation (RAG)** chat application tailored for medical data.
It combines **LangChain** and **LangGraph** for workflow orchestration, **LangSmith** for observability,
and integrates with **FastAPI** for the backend and **Chainlit** for the user interface.

The system uses a **Qdrant vector database** to store and retrieve documents.
Synthetic patient data is generated using **Synthea**.

## ⚙️ Stack

-   **LangChain / LangGraph** workflow orchestration
-   **LangSmith** observability
-   **Qdrant** vector database for retrieval
-   **FastAPI** API backend
-   **Chainlit** chat-based UI
-   **Synthea** synthetic medical data generation
-   **OpenEval** evaluation metrics (helpfulness, correctness, etc.)
-   **Docker & Docker Compose** setup for easy deployment

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/SasCezar/medicalrag.git
cd medicalrag
```

Build and run with Docker:

```bash
docker compose up --build
```

## 📈 Evaluation

The evaluation module runs experiments on synthetic datasets using:

-   **Helpfulness**
-   **Correctness**

Metrics are computed using **LangChain OpenEval** and logged in **LangSmith**.

## 🌐 API

Run FastAPI locally at:

```
http://localhost:8000/docs
```

Endpoints:

-   `POST /chat` → interact with the RAG model

## 💬 UI

Run Chainlit UI:

```
http://localhost:8001
```

## 📊 Workflow Graph

Below is the adaptive workflow graph used in the system:

![RAG Graph](/static/graph.png)
