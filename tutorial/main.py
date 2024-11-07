from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tutorial.simple_graph import SimpleChatbotGraph, run_chatbot_prompt


def run_simple_chatbot():
    run_chatbot_prompt(SimpleChatbotGraph(ChatOpenAI(model="gpt-3.5-turbo")))


if __name__ == "__main__":
    load_dotenv()
    run_simple_chatbot()
