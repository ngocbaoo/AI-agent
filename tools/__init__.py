from .rag import legal_rag_tool
from .compare import compare_logo_similarity_tool, compare_text_similarity_tool
from .search import(
    trademark_search_tool,
    design_search_tool,
    patent_search_tool
)


tools = [
    trademark_search_tool,
    design_search_tool,
    patent_search_tool,
    compare_logo_similarity_tool,
    compare_text_similarity_tool,
    legal_rag_tool
]