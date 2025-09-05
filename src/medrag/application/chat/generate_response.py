from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from loguru import logger

from .workflow.graph import create_workflow_graph


async def get_response(query: str, user_id: str):
    graph_builder = create_workflow_graph()
    logger.info("Running response")
    try:
        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
            graph = graph_builder.compile(checkpointer=checkpointer)
            logger.info("Compiled graph")
            thread_id = {"user_id": user_id}
            config = {
                "configurable": {"thread_id": thread_id, "callbacks": []},
            }
            output = await graph.ainvoke(
                input={"query": query, "user_id": user_id},
                config=config,
            )
            last_message = output["generation"]
            return last_message
    except Exception as e:
        logger.exception("Error with the graph!")
        raise RuntimeError(f"Failed to compile graph: {e}")
