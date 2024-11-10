from typing import List, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    documents: List[str | Document]
    generation: Optional[str]
