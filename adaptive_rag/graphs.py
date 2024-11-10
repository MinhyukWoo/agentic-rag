from langgraph.constants import START, END
from langgraph.graph import StateGraph

from adaptive_rag.edges import (
    route_question,
    decide_to_generate,
    grade_generation_v_documents_and_question,
)
from adaptive_rag.nodes import (
    web_search,
    retrieve,
    grade_documents,
    generate,
    transform_query,
)
from adaptive_rag.states import GraphState


def get_adaptive_rag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
        START, route_question, {"web_search": "web_search", "vectorstore": "retrieve"}
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()
