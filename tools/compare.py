from langchain_core.tools import tool
from thefuzz import fuzz
import numpy as np
from PIL import Image
import io, base64

# Lazy loader cho CLIP
_CLIP_MODEL = None
_CLIP_DEVICE = None

def _load_clip():
    global _CLIP_MODEL, _CLIP_DEVICE
    if _CLIP_MODEL is None:
        print("--- [TOOL LOG] Load 'clip-ViT-B-32' (RAM only) ---")
        from sentence_transformers import SentenceTransformer
        import torch
        _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL = SentenceTransformer("clip-ViT-B-32", device=_CLIP_DEVICE)
        print(f"--- [TOOL LOG] CLIP ready on {_CLIP_DEVICE} ---")
    return _CLIP_MODEL

def _embed_image_b64(b64_str: str) -> np.ndarray:
    if not b64_str:
        raise ValueError("empty base64 image")
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    model = _load_clip()
    emb = model.encode([img], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
    # xoá tham chiếu tạm
    del raw, img
    return emb[0]

def _cosine_scaled(a: np.ndarray, b: np.ndarray) -> float:
    val = float(np.dot(a, b))  # đã normalize
    return max(0.0, min(1.0, (val + 1.0) / 2.0))

@tool
def compare_text_similarity_tool(text1: str, text2: str) -> float:
    """
    Đo tương đồng văn bản 0..1 (fuzz.ratio/100).
    """
    print(f"--- [TOOL LOG] Text compare: '{text1}' vs '{text2}' ---")
    return fuzz.ratio((text1 or "").lower(), (text2 or "").lower()) / 100.0

@tool
def compare_logo_similarity_tool(user_logo_b64: str, candidate_logo_b64: str) -> float:
    """
    So khớp logo bằng CLIP ViT-B/32, input là base64 hai ảnh. Không lưu file.
    Trả về điểm 0..1.
    """
    print("--- [TOOL LOG] CLIP compare (RAM) ---")
    try:
        e1 = _embed_image_b64(user_logo_b64)
        e2 = _embed_image_b64(candidate_logo_b64)
        score = _cosine_scaled(e1, e2)
        print(f"--- [TOOL LOG] CLIP cosine (0..1): {score:.4f} ---")
        # xoá tham chiếu tạm
        del e1, e2
        return round(score, 4)
    except Exception as e:
        print(f"--- [TOOL ERROR] CLIP compare failed: {e}")
        return 0.0
