from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from medrag.application.chat.generate_response import get_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


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
    doc_id: str = "1"


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    try:
        response_text = await get_response(user_text=chat_message.message, user_id=chat_message.doc_id)
        return {"response": response_text}
    except Exception as e:
        logger.exception("Error with the response!")
        raise HTTPException(status_code=500, detail=str(e))
