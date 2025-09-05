from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from medrag.application.chat.generate_response import get_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: e.g., warm caches, load models, etc.
    yield
    # Shutdown: e.g., flush tracers, close pools, etc.


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
    doc_id: str = "1"  # keep this a string to use as thread_id/user_id


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    try:
        # New entry point uses user_text + user_id and expects messages internally
        response_text = await get_response(user_text=chat_message.message, user_id=chat_message.doc_id)
        return {"response": response_text}
    except Exception as e:
        logger.exception("Error with the response!")
        raise HTTPException(status_code=500, detail=str(e))
