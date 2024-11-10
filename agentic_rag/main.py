from dotenv import load_dotenv

from agentic_rag.graphs import get_agentic_rag_graph
from agentic_rag.utils import save_graph_as_png

if __name__ == "__main__":
    load_dotenv()
    save_graph_as_png(get_agentic_rag_graph())
