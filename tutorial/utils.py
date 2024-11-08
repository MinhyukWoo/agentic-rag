from typing import TypedDict, Annotated, Protocol

from langgraph.graph import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class ChatbotGraphInterface(Protocol):
    def stream_graph_updates(self, user_input: str) -> None:
        """Graph 업데이트 스트리밍 메서드를 정의합니다."""
        pass


def run_chatbot_prompt(graph):
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            graph.stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            graph.stream_graph_updates(user_input)
            break
