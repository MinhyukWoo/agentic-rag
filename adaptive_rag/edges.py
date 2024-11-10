from typing import Literal

from dotenv import load_dotenv

from adaptive_rag.models import (
    get_question_router,
    get_hallucination_grader,
    get_answer_grader,
)
from adaptive_rag.states import GraphState

load_dotenv()

question_router = get_question_router()
hallucination_grader = get_hallucination_grader()
answer_grader = get_answer_grader()


def route_question(state: GraphState) -> Literal["vectorstore", "web_search"]:
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        return "web_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"


def decide_to_generate(state: GraphState) -> Literal["transform_query", "generate"]:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        return "generate"


def grade_generation_v_documents_and_question(
    state: GraphState,
) -> Literal["useful", "not useful", "not supported"]:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"
