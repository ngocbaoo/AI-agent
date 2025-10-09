from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from typing import List, Dict


@tool
def patent_search_tool(keyword: str, technical_field: str) -> List[Dict]:
    """
    Quét nhanh tiêu đề và tóm tắt của các bằng sáng chế dựa trên từ khóa và lĩnh vực kỹ thuật.
    """
    print(f"--- [TOOL LOG] Giả lập tra cứu sáng chế '{keyword}' trong lĩnh vực '{technical_field}' ---")
    return [
        {"title": f"Invention related to {keyword}", "abstract": "An abstract about the invention..."},
        {"title": f"A new method for {keyword}", "abstract": "Details of the new method..."},
    ]