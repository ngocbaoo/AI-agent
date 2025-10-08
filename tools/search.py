import os, time, re, io, base64, requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from requests.exceptions import RequestException, JSONDecodeError
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from PIL import Image

from .compare import compare_text_similarity_tool, compare_logo_similarity_tool
load_dotenv()

def _sanitize_for_rsql(name: str) -> str:
    if not name: return ""
    s = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    return re.sub(r'\s+', ' ', s).strip()

_euipo_sandbox_access_token = None
_euipo_sandbox_token_expires_at = 0.0

def _get_euipo_sandbox_access_token() -> Optional[str]:
    global _euipo_sandbox_access_token, _euipo_sandbox_token_expires_at
    if _euipo_sandbox_access_token and time.time() < _euipo_sandbox_token_expires_at:
        return _euipo_sandbox_access_token
    print("--- [TOOL LOG] Fetch EUIPO Sandbox token ---")
    cid = os.environ.get("EU_SANDBOX_ID")
    csec = os.environ.get("EU_SANDBOX_SECRET")
    if not cid or not csec:
        print("--- [TOOL ERROR] Missing EU_SANDBOX_ID/SECRET ---")
        return None
    token_url = "https://auth-sandbox.euipo.europa.eu/oidc/accessToken"
    try:
        r = requests.post(token_url,
                          headers={'Content-Type': 'application/x-www-form-urlencoded'},
                          data={'grant_type': 'client_credentials','client_id': cid,'client_secret': csec,'scope': 'uid'},
                          timeout=10)
        r.raise_for_status()
        data = r.json()
        _euipo_sandbox_access_token = data.get("access_token")
        _euipo_sandbox_token_expires_at = time.time() + data.get("expires_in", 3600) - 60
        print("--- [TOOL LOG] Token OK ---")
        return _euipo_sandbox_access_token
    except RequestException as e:
        print(f"--- [TOOL ERROR] Token error: {e}")
        return None

def _content_type_to_ext(ct: str) -> str:
    ct = (ct or "").lower()
    if "png" in ct: return ".png"
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    if "gif" in ct: return ".gif"
    return ".jpg"

def _to_jpeg_b64(raw_bytes: bytes) -> Optional[str]:
    """
    Chuẩn hoá mọi định dạng ảnh sang JPEG + base64 (in-RAM).
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        # giới hạn chiều lớn nhất 512px để giảm payload
        img.thumbnail((512, 512))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        # cleanup
        buf.close()
        del img
        return b64
    except Exception as e:
        print(f"--- [TOOL WARN] Normalize img failed: {e}")
        return None

def _extract_markimage_b64(candidate: Dict[str, Any], headers: Dict[str, str]) -> Optional[str]:
    """
    Cố gắng lấy ảnh logo EUIPO -> trả về base64 JPEG. Tất cả in-memory.
    """
    app_no = candidate.get("applicationNumber")
    if not app_no: return None

    # 1) Nếu candidate có URL trực tiếp
    mi = candidate.get("markImage") or {}
    url = None
    if isinstance(mi, str) and mi.startswith("http"):
        url = mi
    elif isinstance(mi, dict):
        for k in ("url","uri","href","markImageUri","imageUri","imageUrl"):
            if mi.get(k):
                url = mi[k]; break

    if url:
        try:
            print(f"--- [TOOL LOG] Download markImage URL: {url} (RAM) ---")
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            return _to_jpeg_b64(r.content)
        except Exception as e:
            print(f"--- [TOOL WARN] Download failed: {e}")

    # 2) Thử gọi detail xem có base64/content
    base = "https://api-sandbox.euipo.europa.eu/trademark-search/trademarks"
    try:
        d = requests.get(f"{base}/{app_no}", headers=headers, timeout=20)
        if d.status_code == 200:
            jd = d.json()
            img = jd.get("markImage") or {}
            # base64 variant?
            for k in ("content","base64","imageBase64","imageContent"):
                if isinstance(img, dict) and isinstance(img.get(k), str) and len(img.get(k)) > 100:
                    try:
                        raw = base64.b64decode(img[k])
                        return _to_jpeg_b64(raw)
                    except Exception as e:
                        print(f"--- [TOOL WARN] b64->jpeg failed: {e}")
                        break
            # link ở detail?
            for k in ("url","uri","href","markImageUri","imageUri","imageUrl"):
                if isinstance(img, dict) and img.get(k):
                    try:
                        url = img[k]
                        print(f"--- [TOOL LOG] Download markImage(detail): {url} ---")
                        r = requests.get(url, headers=headers, timeout=15)
                        r.raise_for_status()
                        return _to_jpeg_b64(r.content)
                    except Exception as e:
                        print(f"--- [TOOL WARN] Download(detail) failed: {e}")
                        break
        else:
            print(f"--- [TOOL WARN] Detail {app_no} HTTP {d.status_code}")
    except Exception as e:
        print(f"--- [TOOL WARN] Detail error: {e}")
    return None

# ===== Tool =====
class TrademarkSearchInput(BaseModel):
    name: str = Field(description="Tên nhãn hiệu cần tra cứu.")
    nice_class: Optional[int] = Field(default=None)
    threshold: Optional[float] = Field(default=None)
    user_logo_b64: Optional[str] = Field(default=None, description="(Optional) Base64 JPEG/PNG logo của người dùng.")

@tool(args_schema=TrademarkSearchInput)
def trademark_search_tool(name: str, nice_class: Optional[int] = None,
                          threshold: Optional[float] = 0.85,
                          user_logo_b64: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Tra cứu + chấm điểm (tên + logo nếu có). Không lưu file.
    Trả về Top-5 theo combined_score.
    """
    original_name = name
    sanitized_name = _sanitize_for_rsql(name)
    print(f"--- [SEARCH LOG] Query='{sanitized_name}' class={nice_class} ---")
    if not sanitized_name:
        return [{"error": "Tên sau khi làm sạch rỗng."}]

    token = _get_euipo_sandbox_access_token()
    if not token:
        return [{"error": "Xác thực EUIPO Sandbox thất bại."}]
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}',
        'X-IBM-Client-Id': os.environ.get("EU_SANDBOX_ID")
    }
    api = "https://api-sandbox.euipo.europa.eu/trademark-search/trademarks"
    # Lấy cả marks có hình: KHÔNG giới hạn markFeature==WORD
    q = f"wordMarkSpecification.verbalElement==*{sanitized_name}*"
    if nice_class:
        q += f" and niceClasses=={nice_class}"
    params = {"query": q, "size": 25}

    try:
        r = requests.get(api, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("trademarks", [])
    except (JSONDecodeError, RequestException) as e:
        return [{"error": f"API error: {e}"}]

    if not candidates:
        return [{"message": f"Không tìm thấy chứa '{sanitized_name}'."}]

    thr = threshold if threshold is not None else 0.85
    filtered: List[Dict[str, Any]] = []
    for c in candidates:
        cand_name = (c.get("wordMarkSpecification") or {}).get("verbalElement", "")
        if not cand_name: continue
        s1 = compare_text_similarity_tool.invoke({"text1": original_name, "text2": cand_name})
        s2 = compare_text_similarity_tool.invoke({"text1": sanitized_name, "text2": _sanitize_for_rsql(cand_name)})
        name_score = max(s1, s2)
        print(f"--- [SCORE LOG] Name '{original_name}' vs '{cand_name}': {name_score:.3f}")
        if name_score >= thr:
            c["similarity_score"] = round(name_score, 3)
            filtered.append(c)

    if not filtered:
        return [{"message": f"Không ứng viên đạt ngưỡng >= {thr}"}]

    # Chuẩn hoá user logo sang JPEG b64 (nếu có)
    user_b64 = None
    if user_logo_b64:
        try:
            # Cho phép input là png/jpg b64 bất kỳ → chuẩn hoá lại một lần
            raw = base64.b64decode(user_logo_b64)
            user_b64 = _to_jpeg_b64(raw)
            del raw
        except Exception as e:
            print(f"--- [TOOL WARN] User logo invalid base64: {e}")
            user_b64 = None

    # Lấy ảnh EUIPO (RAM) + CLIP
    for c in filtered:
        c["logo_similarity"] = None
        c["markImageBase64"] = None  # để hiển thị (nếu cần)
        if user_b64:
            cand_b64 = _extract_markimage_b64(c, headers)
            if cand_b64:
                try:
                    c["markImageBase64"] = cand_b64  # dùng để render preview nếu UI cần
                    ls = compare_logo_similarity_tool.invoke({
                        "user_logo_b64": user_b64,
                        "candidate_logo_b64": cand_b64
                    })
                    c["logo_similarity"] = float(ls)
                    print(f"--- [CLIP LOG] {c.get('applicationNumber')} logo_sim={c['logo_similarity']}")
                except Exception as e:
                    print(f"--- [TOOL WARN] CLIP compare failed: {e}")

        # combined score
        if c["logo_similarity"] is not None:
            c["combined_score"] = round(0.5 * c["similarity_score"] + 0.5 * c["logo_similarity"], 4)
        else:
            c["combined_score"] = float(c["similarity_score"])

    filtered.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    top5 = filtered[:5]

    # Xoá reference lớn khỏi RAM sau khi trả về (Python GC sẽ làm phần còn lại)
    if user_b64: del user_b64
    print(f"--- [SEARCH LOG] Done. Top-{len(top5)} ---")
    return top5

    
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