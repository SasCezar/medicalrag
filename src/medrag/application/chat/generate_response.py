from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger

from .workflow.graph import graph


async def get_response(user_text: str, user_id: str) -> str:
    """
    Single-turn entry that plugs into the chat-based graph.
    You pass the user's text, it appends to the message list internally
    and returns the last AI response as a string.
    """
    # graph_builder = create_workflow_graph()
    logger.info("Running response")

    try:
        # For persistence, use a file path, e.g. 'checkpoint.db'.
        # async with AsyncSqliteSaver.from_conn_string("checkpoint.db") as checkpointer:
        # graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("Compiled graph")

        # Use a stable, hashable thread id (string is simplest)
        config = {"configurable": {"thread_id": str(user_id)}}

        # Chat-based input: wrap user text as a HumanMessage
        state_out = await graph.ainvoke(
            input={"messages": [HumanMessage(content=user_text)]},
            config=config,
        )

        # Extract last AI reply
        messages: list[BaseMessage] = state_out.get("messages", []) or []
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai and last_ai.content:
            return last_ai.content

        # Fallback if your nodes also set 'agent_output'
        agent_output = state_out.get("agent_output")
        if agent_output:
            return agent_output

        logger.warning("No AIMessage or agent_output found in graph output.")
        return ""

    except Exception as e:
        logger.exception("Error with the graph!")
        raise RuntimeError(f"Failed to run graph: {e}")
