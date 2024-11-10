from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from agentic_rag.utils import get_retriever_tools
from tutorial.utils import State


def agent(state: State):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")
    model = model.bind_tools(get_retriever_tools())
    response = model.invoke(messages)
    return {"messages": [response]}


def rewrite(state: State):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question:"""
        )
    ]

    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state: State):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    messages = state["messages"]
    question = messages[0].content
    last_messages = messages[-1]

    docs = last_messages.content

    prompt = hub.pull("rlm/rag-prompt")
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")

    def format_docs(docs):
        return "\n\n".join(doc.page_contents for doc in docs)

    rag_chain = prompt | model | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}
