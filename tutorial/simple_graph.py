from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from tutorial.utils import State, ChatbotGraphInterface


class SimpleChatbotGraph(ChatbotGraphInterface):
    def __init__(self, llm: ChatOpenAI):
        self.graph = SimpleChatbotGraph._build_graph(llm)

    @staticmethod
    def _build_graph(llm) -> CompiledStateGraph:
        def chatbot(state: State):
            return {"messages": [llm.invoke(state["messages"])]}

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        return graph_builder.compile()

    def stream_graph_updates(self, user_input: str) -> None:
        for event in self.graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
