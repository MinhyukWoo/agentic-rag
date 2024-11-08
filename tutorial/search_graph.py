import json

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import ToolMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from tutorial.utils import State, ChatbotGraphInterface


def get_tavily_search_tools():
    tool = TavilySearchResults(max_results=2)
    tool.invoke("What's a 'node' in LangGraph?")
    return [tool]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


class SearchChatbotGraph(ChatbotGraphInterface):
    def __init__(self, llm: ChatOpenAI):
        llm_with_tools = llm.bind_tools(get_tavily_search_tools())
        self.graph = SearchChatbotGraph._build_graph(llm_with_tools)

    @staticmethod
    def _build_graph(llm) -> CompiledStateGraph:
        def chatbot(state: State):
            return {"messages": [llm.invoke(state["messages"])]}

        tool_node = BasicToolNode(tools=get_tavily_search_tools())

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_conditional_edges(
            "chatbot", route_tools, {"tools": "tools", END: END}
        )
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")
        return graph_builder.compile()

    def stream_graph_updates(self, user_input: str):
        for event in self.graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
