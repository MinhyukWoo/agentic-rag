from langgraph.graph.state import CompiledStateGraph


def save_graph_as_png(graph: CompiledStateGraph, filename="graph_output"):
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(f"{filename}.png", "wb") as f:
            f.write(png_data)
        print("PNG 파일이 저장되었습니다.")
    except Exception as e:
        print("이미지 생성 중 오류가 발생했습니다:", e)
