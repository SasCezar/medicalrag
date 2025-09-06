from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from .workflow.graph import create_workflow_graph

checkpointer = MemorySaver()
graph = create_workflow_graph().compile(checkpointer=checkpointer)


async def get_response(user_text: str, user_id: str) -> str:
    logger.info(f"Running pipeline for query `{user_text}`")

    try:
        config = {"configurable": {"thread_id": str(user_id)}}

        state_out = await graph.ainvoke(
            input={"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

        messages: list[BaseMessage] = state_out.get("messages", [])
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai and last_ai.content:
            return last_ai.content

        agent_output = state_out.get("agent_output")
        if agent_output:
            return agent_output

        logger.warning("No AIMessage or agent_output found in graph output.")
        return ""

    except Exception as e:
        logger.exception("Error with the graph!")
        raise RuntimeError(f"Failed to run graph: {e}")
