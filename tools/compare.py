from langchain_core.tools import tool
from thefuzz import fuzz
import random

@tool
def compare_text_similarity_tool(text1: str, text2: str) -> float:
    """
    Đo lường mức độ tương đồng giữa hai chuỗi văn bản (0.0 đến 1.0).
    """
    print(f"--- [TOOL LOG] So sánh văn bản: '{text1}' vs '{text2}' ---")
    similarity_ratio = fuzz.ratio(text1.lower(), text2.lower()) / 100.0
    return similarity_ratio

@tool
def compare_logo_similarity_tool(logo_path1: str, logo_path2: str) -> float:
    """
    Đo lường mức độ tương đồng hình ảnh giữa hai logo (0.0 đến 1.0).
    Hàm này phải chạy hoàn toàn cục bộ để đảm bảo bảo mật.
    """
    print(f"--- [TOOL LOG] Giả lập so sánh logo: '{logo_path1}' vs '{logo_path2}' ---")
    return round(random.uniform(0.6, 0.9), 2)