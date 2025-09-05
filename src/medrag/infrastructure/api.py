from contextlib import asynccontextmanager

# from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from medrag.application.chat.generate_response import get_response

# from .opik_utils import configure

# configure()
# load_dotenv(find_dotenv(filename=".env", usecwd=True))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the API."""
    # Startup code (if any) goes here
    yield
    # Shutdown code goes here
    # opik_tracer = OpikTracer()
    # opik_tracer.flush()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str
    doc_id: str = 1


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    try:
        response = await get_response(query=chat_message.message, user_id=chat_message.doc_id)
        return {"response": response}
    except Exception as e:
        logger.exception("Error with the respone!")
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
