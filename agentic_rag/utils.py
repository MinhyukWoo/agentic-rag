from typing import TypedDict, Annotated, Sequence

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


def get_retriever_tools():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )

    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits, collection_name="rag-chroma", embedding=OpenAIEmbeddings()
    )

    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
    )
    return [retriever_tool]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def save_graph_as_png(graph: CompiledStateGraph, filename="graph_output"):
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(f"{filename}.png", "wb") as f:
            f.write(png_data)
        print("PNG 파일이 저장되었습니다.")
    except Exception as e:
        print("이미지 생성 중 오류가 발생했습니다:", e)
