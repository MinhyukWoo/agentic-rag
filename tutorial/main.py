from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tutorial.search_graph import SearchChatbotGraph
from tutorial.simple_graph import SimpleChatbotGraph
from tutorial.utils import run_chatbot_prompt


def run_simple_chatbot():
    run_chatbot_prompt(SimpleChatbotGraph(ChatOpenAI(model="gpt-3.5-turbo")))


def save_graph_as_png(graph, filename="graph_output"):
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(f"{filename}.png", "wb") as f:
            f.write(png_data)
        print("PNG 파일이 저장되었습니다.")
    except Exception as e:
        print("이미지 생성 중 오류가 발생했습니다:", e)


def run_search_chatbot():
    graph = SearchChatbotGraph(ChatOpenAI(model="gpt-3.5-turbo"))
    save_graph_as_png(graph.graph)
    run_chatbot_prompt(graph)


if __name__ == "__main__":
    load_dotenv()
    run_search_chatbot()
