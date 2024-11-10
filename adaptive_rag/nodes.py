from dotenv import load_dotenv
from langchain_core.documents import Document

from adaptive_rag.models import (
    get_retriever,
    get_rag_chain,
    get_retrieval_grader,
    get_question_rewriter,
    get_web_search_tool,
)
from adaptive_rag.states import GraphState

load_dotenv()
retriever = get_retriever()
rag_chain = get_rag_chain()
retrieval_grader = get_retrieval_grader()
question_rewriter = get_question_rewriter()
web_search_tool = get_web_search_tool()


def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "generation": None}


def generate(state: GraphState) -> GraphState:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for document in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(document)
        else:
            continue
    return {"documents": filtered_docs, "question": question, "generation": None}


def transform_query(state: GraphState) -> GraphState:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    question = state["question"]
    documents = state["documents"]

    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question, "generation": None}


def web_search(state: GraphState) -> GraphState:
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    question = state["question"]

    documents = web_search_tool.invoke({"query": question})
    web_content = "\n".join([document["content"] for document in documents])
    web_results = [Document(page_content=web_content)]

    return {"documents": web_results, "question": question, "generation": None}
