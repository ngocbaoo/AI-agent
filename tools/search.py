import os
import time
import re
import requests
from functools import lru_cache
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from requests.exceptions import RequestException, JSONDecodeError
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

from .compare import compare_logo_similarity_tool, compare_text_similarity_tool
load_dotenv()

def _sanitize_for_rsql(name: str) -> str:
    """
    Loại bỏ các ký tự đặc biệt không hợp lệ cho truy vấn RSQL 
    và chuẩn hóa khoảng trắng.
    """
    if not name:
        return ""
    # Chỉ giữ lại chữ cái, số, và khoảng trắng
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    # Thay thế nhiều khoảng trắng bằng một khoảng trắng duy nhất
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized

# -- Trademark search tool ---
_euipo_sandbox_access_token: str | None = None
_euipo_sandbox_token_expires_at: float = 0.0

def _get_euipo_sandbox_access_token() -> str | None:
    global _euipo_sandbox_access_token, _euipo_sandbox_token_expires_at
    if _euipo_sandbox_access_token and time.time() < _euipo_sandbox_token_expires_at:
        return _euipo_sandbox_access_token
    print("--- [TOOL LOG] Yêu cầu token mới từ API Sandbox (RSQL)... ---")
    client_id = os.environ.get("EU_SANDBOX_ID")
    client_secret = os.environ.get("EU_SANDBOX_SECRET")
    if not client_id or not client_secret:
        print("--- [TOOL ERROR] EU_SANDBOX_ID hoặc SECRET chưa được thiết lập. ---")
        return None
    token_url = "https://auth-sandbox.euipo.europa.eu/oidc/accessToken"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'client_credentials', 'client_id': client_id, 'client_secret': client_secret, 'scope': 'uid'}
    try:
        response = requests.post(token_url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        access_token = response_data.get("access_token")
        expires_in_seconds = response_data.get("expires_in", 3600)
        _euipo_sandbox_access_token = access_token
        _euipo_sandbox_token_expires_at = time.time() + expires_in_seconds - 60
        print("--- [TOOL LOG] Đã lấy token API Sandbox (RSQL) thành công. ---")
        return _euipo_sandbox_access_token
    except RequestException as e:
        print(f"--- [TOOL ERROR] Không thể lấy token (RSQL): {e.response.text if e.response else e} ---")
        return None
    

# --- Công cụ Tra cứu API (API Search Tools) ---
class TrademarkSearchInput(BaseModel):
    name: str = Field(description="Tên nhãn hiệu cần tra cứu.")
    nice_class: Optional[int] = Field(default=None, description="Nhóm Nice (tùy chọn).")
    threshold: Optional[float] = Field(default=None, description="Ngưỡng tương đồng từ 0.0 đến 1.0. Nếu được cung cấp, tool sẽ thực hiện tìm kiếm gần đúng.")
    logo_path: Optional[str] = Field(default=None, description="Đường dẫn hoặc URL đến logo của người dùng để so sánh trực quan.")


def _collect_candidate_logo_urls(candidate: Dict[str, Any]) -> List[str]:
    """Cố gắng trích xuất URL logo từ phản hồi EUIPO (sandbox)."""

    def _walk(value: Any, hint: str = "") -> List[str]:
        urls: List[str] = []
        if isinstance(value, dict):
            for key, nested in value.items():
                combined_hint = f"{hint}.{key}" if hint else key
                urls.extend(_walk(nested, combined_hint))
        elif isinstance(value, list):
            for item in value:
                urls.extend(_walk(item, hint))
        elif isinstance(value, str):
            lowered_hint = hint.lower()
            if value.startswith("http"):
                if any(token in lowered_hint for token in ("image", "logo", "picture", "representation")):
                    urls.append(value)
                else:
                    stripped = value.split("?")[0].lower()
                    if stripped.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                        urls.append(value)
        return urls

    collected = _walk(candidate)
    seen = set()
    unique_urls: List[str] = []
    for url in collected:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


@lru_cache(maxsize=64)
def _logo_similarity_cached(user_logo: str, candidate_logo: str) -> float:
    """Cache kết quả so sánh logo để giảm số lần tải xuống."""
    return compare_logo_similarity_tool.invoke({"logo_path1": user_logo, "logo_path2": candidate_logo})


@tool(args_schema=TrademarkSearchInput)
def trademark_search_tool(
    name: str,
    nice_class: Optional[int] = None,
    threshold: Optional[float] = 0.85,
    logo_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Tra cứu nhãn hiệu và cung cấp điểm tương đồng về văn bản lẫn hình ảnh.
    """
    original_name = name
    sanitized_name = _sanitize_for_rsql(name)

    print(f"--- [SEARCH LOG] Bắt đầu tra cứu cho '{original_name}' (đã làm sạch thành '{sanitized_name}') ---")

    if not sanitized_name:
        return [{"error": "Tên nhãn hiệu sau khi làm sạch bị rỗng, không thể tìm kiếm."}]

    access_token = _get_euipo_sandbox_access_token()
    if not access_token:
        return [{"error": "Xác thực EUIPO Sandbox (RSQL) thất bại."}]
    api_url = "https://api-sandbox.euipo.europa.eu/trademark-search/trademarks"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}', 'X-IBM-Client-Id': os.environ.get("EU_SANDBOX_ID")}
    query = f"markFeature==WORD and wordMarkSpecification.verbalElement==*{sanitized_name}*"
    if nice_class:
        query += f" and niceClasses=={nice_class}"
    params = {"query": query, "size": 10}
    try:
        r = requests.get(api_url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        broad_search_results = data.get("trademarks", [])
    except (JSONDecodeError, RequestException) as e:
        return [{"error": f"(Sandbox RSQL) Lỗi khi gọi API: {e}"}]
    if not broad_search_results:
        return [{"message": f"Không tìm thấy nhãn hiệu nào chứa '{sanitized_name}'."}]

    effective_threshold = threshold if threshold is not None else 0.85
    print(f"--- [SEARCH LOG] Tìm thấy {len(broad_search_results)} ứng cử viên. Bắt đầu lọc đa chiều với ngưỡng {effective_threshold}... ---")

    final_results: List[Dict[str, Any]] = []
    for candidate in broad_search_results:
        candidate_name = candidate.get("wordMarkSpecification", {}).get("verbalElement", "")
        if not candidate_name:
            continue

        sanitized_candidate_name = _sanitize_for_rsql(candidate_name)

        score_original = compare_text_similarity_tool.invoke({"text1": original_name, "text2": candidate_name})
        score_sanitized = compare_text_similarity_tool.invoke({"text1": sanitized_name, "text2": sanitized_candidate_name})
        text_score = max(score_original, score_sanitized)
        candidate["text_similarity_score"] = round(text_score, 2)

        logo_score = 0.0
        logo_source = None
        if logo_path:
            for url in _collect_candidate_logo_urls(candidate):
                try:
                    score = _logo_similarity_cached(logo_path, url)
                except Exception as exc:
                    print(f"--- [SCORE ERROR] Không thể so sánh logo với '{url}': {exc}")
                    continue
                if score > logo_score:
                    logo_score = score
                    logo_source = url

        if logo_score:
            candidate["logo_similarity_score"] = round(logo_score, 2)
            candidate["logo_source"] = logo_source

        overall_score = max(text_score, logo_score)

        print(
            "--- [SCORE LOG] '%s' vs '%s': Original Score=%.2f, Sanitized Score=%.2f, Logo Score=%.2f -> Overall=%.2f"
            % (original_name, candidate_name, score_original, score_sanitized, logo_score, overall_score)
        )

        if overall_score >= effective_threshold:
            candidate["similarity_score"] = round(overall_score, 2)
            final_results.append(candidate)

    if final_results:
        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return final_results

    if logo_path:
        logo_ranked = [c for c in broad_search_results if c.get("logo_similarity_score")]
        if logo_ranked:
            logo_ranked.sort(key=lambda x: x["logo_similarity_score"], reverse=True)
            return logo_ranked[:5]

    return [{"message": f"Không tìm thấy nhãn hiệu nào đạt ngưỡng tương đồng >= {effective_threshold} với '{original_name}'"}]

@tool
def design_search_tool(keyword: str, locarno_class: str) -> List[Dict]:
    """
    Tìm kiếm các kiểu dáng công nghiệp dựa trên từ khóa và phân loại Locarno.
    """
    print(f"--- [TOOL LOG] Giả lập tra cứu kiểu dáng '{keyword}', nhóm Locarno {locarno_class} ---")
    return [
        {"design_name": f"Cool_{keyword}_Design", "designer": "Designer A"},
        {"design_name": f"Creative_{keyword}_Shape", "designer": "Designer B"}
    ]

@tool
def patent_search_tool(keyword: str, technical_field: str) -> List[Dict]:
    """
    Quét nhanh tiêu đề và tóm tắt của các bằng sáng chế dựa trên từ khóa và lĩnh vực kỹ thuật.
    """
    print(f"--- [TOOL LOG] Giả lập tra cứu sáng chế '{keyword}' trong lĩnh vực '{technical_field}' ---")
    return [
        {"title": f"Invention related to {keyword}", "abstract": "An abstract about the invention..."},
        {"title": f"A new method for {keyword}", "abstract": "Details of the new method..."}
    ]