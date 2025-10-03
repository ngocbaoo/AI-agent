from langchain_core.tools import tool
from thefuzz import fuzz
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import base64
import io 
from typing import List

try:
    print("--- INFO: Đang tải mô hình AI để phân tích logo (có thể mất một lúc)... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = SentenceTransformer('clip-ViT-B-32', device=device)
    print(f"--- INFO: Mô hình AI đã sẵn sàng trên '{device}'. ---")
except Exception as e:
    print(f"--- ERROR: Lỗi khi tải mô hình AI: {e} ---")
    clip_model = None

@tool
def compare_text_similarity_tool(text1: str, text2: str) -> float:
    """
    Đo lường mức độ tương đồng giữa hai chuỗi văn bản (0.0 đến 1.0).
    """
    print(f"--- [TOOL LOG] So sánh văn bản: '{text1}' vs '{text2}' ---")
    similarity_ratio = fuzz.ratio(text1.lower(), text2.lower()) / 100.0
    return similarity_ratio

@tool
def compare_logo_similarity_tool(logo_base64_1: str, logo_base64_2: str) -> float:
    """
    Đo lường mức độ tương đồng hình ảnh giữa hai logo 
    Đầu vào là hai chuỗi văn bản đã được mã hóa Base64.
    Trả về điểm từ 0.0 (khác nhau) đến 1.0 (giống nhau).
    """
    if not clip_model:
        print("--- [TOOL ERROR] Mô hình AI chưa được tải, không thể so sánh logo. ---")
        return 0.0

    print(f"--- [TOOL LOG] Bắt đầu so sánh 2 logo bằng AI (in-memory)... ---")
    
    try:
        # Giải mã chuỗi 1 từ Base64 về dạng bytes và mở ảnh
        image_bytes_1 = base64.b64decode(logo_base64_1)
        image_1 = Image.open(io.BytesIO(image_bytes_1))

        # Giải mã chuỗi 2
        image_bytes_2 = base64.b64decode(logo_base64_2)
        image_2 = Image.open(io.BytesIO(image_bytes_2))

        # Dùng mô hình AI để tạo "vector ý nghĩa" (embedding) cho mỗi ảnh
        embeddings = clip_model.encode([image_1, image_2], convert_to_tensor=True)

        # Tính toán độ tương đồng Cosine giữa hai vector
        from sentence_transformers.util import cos_sim
        similarity = cos_sim(embeddings[0], embeddings[1]).item()
        
        # Chuẩn hóa kết quả về thang điểm từ 0.0 đến 1.0
        normalized_similarity = (similarity + 1) / 2
        
        score = round(normalized_similarity, 2)
        print(f"--- [TOOL LOG] Điểm tương đồng logo (AI): {score} ---")
        return score

    except Exception as e:
        print(f"--- [TOOL ERROR] Lỗi trong quá trình so sánh logo bằng AI: {e} ---")
        return 0.0