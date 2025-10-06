from __future__ import annotations

import io
from typing import Optional

import numpy as np
import requests
from PIL import Image
from langchain_core.tools import tool
from requests.exceptions import RequestException
from thefuzz import fuzz


def _load_image(path_or_url: str) -> Optional[Image.Image]:
    """Tải ảnh từ đường dẫn cục bộ hoặc URL."""
    try:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            response = requests.get(path_or_url, timeout=15)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            image = Image.open(path_or_url)
        return image.convert("L")
    except (FileNotFoundError, RequestException, OSError) as exc:
        print(f"--- [TOOL ERROR] Không thể tải ảnh '{path_or_url}': {exc}")
        return None


def _difference_hash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """Tính difference hash (dHash) cho một ảnh grayscale."""
    if hasattr(Image, "Resampling"):
        resample_filter = Image.Resampling.LANCZOS
    else:  # Pillow < 10
        resample_filter = Image.LANCZOS

    resized = image.resize((hash_size + 1, hash_size), resample=resample_filter)
    pixels = np.asarray(resized, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def _hash_similarity(hash_a: np.ndarray, hash_b: np.ndarray) -> float:
    """Trả về mức độ tương đồng (0.0 - 1.0) giữa hai vector hash."""
    if hash_a.size != hash_b.size or hash_a.size == 0:
        return 0.0
    hamming_distance = float(np.count_nonzero(hash_a != hash_b))
    return 1.0 - (hamming_distance / hash_a.size)

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
    Nếu một trong hai ảnh không thể tải được, trả về 0.0.
    """
    print(f"--- [TOOL LOG] So sánh logo: '{logo_path1}' vs '{logo_path2}' ---")
    image_a = _load_image(logo_path1)
    image_b = _load_image(logo_path2)

    if image_a is None or image_b is None:
        return 0.0

    hash_a = _difference_hash(image_a)
    hash_b = _difference_hash(image_b)
    similarity = _hash_similarity(hash_a, hash_b)
    return round(float(similarity), 2)
