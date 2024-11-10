from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

from adaptive_rag.graphs import get_adaptive_rag_graph


def get_answer_for(question, graph: CompiledStateGraph):
    inputs = {"question": question}
    for output in graph.stream(inputs):
        for key, value in output.items():
            pass
    return value["generation"]


if __name__ == "__main__":
    load_dotenv()

    print(
        get_answer_for(
            "비트 코인에 대해서 알려줘.",
            get_adaptive_rag_graph(),
        )
    )

    print(
        get_answer_for(
            "AI Agent에서 메모리 타입을 모두 알려줘.",
            get_adaptive_rag_graph(),
        )
    )

    print(
        get_answer_for(
            "Self-reflection의 저자가 누구인지 알려줘",
            get_adaptive_rag_graph(),
        )
    )
