import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage

from medrag.application.chat.workflow.graph import graph


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    async for node_msg, metadata in graph.astream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        if isinstance(node_msg, AIMessage) and metadata.get("langgraph_node") == "generate":
            if node_msg.content:
                await final_answer.stream_token(node_msg.content)

    await final_answer.send()
