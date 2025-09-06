FROM python:3.13-slim AS base
WORKDIR /medrag

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* README.md ./

RUN uv pip install --system --no-cache .

COPY . .

RUN uv pip install --system --no-cache -e .


# FastAPI image
FROM base AS fastapi
EXPOSE 8000
CMD ["uvicorn", "medrag.infrastructure.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Chainlit image
FROM base AS chainlit
ENV CHAINLIT_APP=src/medrag/infrastructure/chainlit.py
EXPOSE 8001
CMD ["chainlit", "run", "src/medrag/infrastructure/chainlit.py", "--host", "0.0.0.0", "--port", "8001", "--no-cache"]
