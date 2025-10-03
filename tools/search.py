import os
import time
import re
import requests
import base64
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from requests.exceptions import RequestException, JSONDecodeError
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

from .compare import compare_text_similarity_tool, compare_logo_similarity_tool
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

def _get_trademark_image_base64(trademark_id: str) -> Optional[str]:
    """
    Hàm nội bộ để tải logo của một nhãn hiệu và trả về chuỗi Base64.
    """
    print(f"--- [SUB-TOOL LOG] Đang tải logo cho ID: {trademark_id} ---")
    image_api_url = f"https://api-sandbox.euipo.europa.eu/trademark-search/trademarks/{trademark_id}/image"
    access_token = _get_euipo_sandbox_access_token()
    if not access_token: return None
    
    headers = {'Authorization': f'Bearer {access_token}', 'X-IBM-Client-Id': os.environ.get("EU_SANDBOX_ID")}
    try:
        response = requests.get(image_api_url, headers=headers, timeout=30)
        response.raise_for_status()
        if 'image' not in response.headers.get('content-type', ''):
            print(f"--- [SUB-TOOL WARN] API không trả về hình ảnh cho ID {trademark_id}.")
            return None
        return base64.b64encode(response.content).decode('utf-8')
    except RequestException as e:
        print(f"--- [SUB-TOOL ERROR] Lỗi khi tải ảnh cho ID {trademark_id}: {e}")
        return None

# --- Công cụ Tra cứu API (API Search Tools) ---
class TrademarkSearchInput(BaseModel):
    name: str = Field(description="Tên nhãn hiệu cần tra cứu.")
    nice_class: Optional[int] = Field(default=None, description="Nhóm Nice (tùy chọn).")
    threshold: Optional[float] = Field(default=0.85, description="Ngưỡng tương đồng văn bản.")
    logo_base64: Optional[str] = Field(default=None, description="Chuỗi Base64 của logo người dùng (tùy chọn).")

    
@tool(args_schema=TrademarkSearchInput)
def trademark_search_tool(name: str, nice_class: Optional[int] = None, threshold: Optional[float] = 0.85, logo_base64: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Tra cứu nhãn hiệu, sử dụng phương pháp "so sánh đa chiều" để xử lý các
    trường hợp cố tình viết sai chính tả bằng ký tự đặc biệt.
    """
    original_name = name
    sanitized_name = _sanitize_for_rsql(name)

    print(f"--- [SEARCH LOG] Bắt đầu tra cứu cho '{original_name}' (đã làm sạch thành '{sanitized_name}') ---")
    
    if not sanitized_name:
        return [{"error": "Tên nhãn hiệu sau khi làm sạch bị rỗng, không thể tìm kiếm."}]

    # --- BƯỚC 1: TRA CỨU MỞ RỘNG  ---
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

    # --- BƯỚC 2: LỌC KẾT QUẢ BẰNG "SO SÁNH ĐA CHIỀU" ---
    effective_threshold = threshold if threshold is not None else 0.85
    print(f"--- [SEARCH LOG] Tìm thấy {len(broad_search_results)} ứng cử viên. Bắt đầu lọc đa chiều với ngưỡng {effective_threshold}... ---")

    final_results = []
    for candidate in broad_search_results:
        candidate_name = candidate.get("wordMarkSpecification", {}).get("verbalElement", "")
        if not candidate_name:
            continue
        
        # "Làm sạch" cả tên của ứng cử viên
        sanitized_candidate_name = _sanitize_for_rsql(candidate_name)

        # So sánh 1: Nguyên bản vs. Nguyên bản
        score_original = compare_text_similarity_tool.invoke({"text1": original_name, "text2": candidate_name})
        
        # So sánh 2: "Lõi" vs. "Lõi"
        score_sanitized = compare_text_similarity_tool.invoke({"text1": sanitized_name, "text2": sanitized_candidate_name})

        # Lấy điểm cao nhất làm điểm cuối cùng
        final_score = max(score_original, score_sanitized)
        
        print(f"--- [SCORE LOG] '{original_name}' vs '{candidate_name}': Original Score={score_original:.2f}, Sanitized Score={score_sanitized:.2f} -> Final Score={final_score:.2f}")

        if final_score >= effective_threshold:
            candidate["similarity_score"] = round(final_score, 2)
            final_results.append(candidate)

    if not final_results:
        return [{"message": f"Không tìm thấy nhãn hiệu nào đạt ngưỡng tương đồng >= {effective_threshold} với '{original_name}'"}]


    # --- BƯỚC 3 - LẤY, SO SÁNH, VÀ TRẢ VỀ DỮ LIỆU LOGO ---
    if not logo_base64:
        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return final_results
    
    print(f"--- [SEARCH LOG] Bắt đầu quá trình tải và so sánh logo... ---")
    
    for result in final_results:
        trademark_id = result.get("applicationNumber")
        if not trademark_id:
            result["logo_similarity_score"] = 0.0
            continue

        candidate_logo_b64 = _get_trademark_image_base64(trademark_id)

        if candidate_logo_b64:
            logo_score = compare_logo_similarity_tool.invoke({
                "logo_base64_1": logo_base64,
                "logo_base64_2": candidate_logo_b64
            })
            result["logo_similarity_score"] = logo_score
            result["retrieved_logo_base64"] = candidate_logo_b64
        else:
            result["logo_similarity_score"] = 0.0

    final_results.sort(key=lambda x: (x.get("logo_similarity_score", 0.0), x["similarity_score"]), reverse=True)
    return final_results
    
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