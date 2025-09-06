import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from medrag.application.chat.workflow.graph import create_workflow_graph

checkpointer = MemorySaver()
graph = create_workflow_graph().compile(checkpointer=checkpointer)


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    async for msg, metadata in graph.astream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        if msg.content and not isinstance(msg, HumanMessage) and metadata["langgraph_node"] == "generate":
            await final_answer.stream_token(msg.content)

    await final_answer.send()
