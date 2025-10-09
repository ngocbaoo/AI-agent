import os, time, re, io, base64, requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from requests.exceptions import RequestException, JSONDecodeError
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from PIL import Image, ImageFile, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()

from .compare import compare_text_similarity_tool, compare_logo_similarity_tool

# Context để app.py set ảnh (tránh nhét base64 vào prompt)
USER_LOGO_B64_CTX: Optional[str] = None


# ===================== Base64 helpers =====================
def decode_any_base64(s: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Trả về (raw_bytes, mime_from_prefix|None).
    - Nhận cả chuỗi 'data:*;base64,AAAA...' lẫn chuỗi b64 thuần.
    - Làm sạch khoảng trắng, URL-safe, padding.
    """
    if not s:
        return None, None
    s = s.strip().strip('"').strip("'")
    mime = None
    if s.startswith("data:"):
        comma = s.find(",")
        if comma == -1:
            return None, None
        header = s[5:comma]              # ví dụ: image/png;base64
        payload = s[comma + 1 :]
        semi = header.find(";")
        mime = (header[:semi] if semi != -1 else header).lower()
    else:
        payload = s
    # loại bỏ khoảng trắng/newlines; một số nguồn thay '+' thành ' '
    payload = re.sub(r"\s+", "", payload).replace(" ", "+")
    # padding bội số 4
    pad = (-len(payload)) % 4
    if pad:
        payload += "=" * pad
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception:
        payload2 = payload.replace("-", "+").replace("_", "/")
        pad = (-len(payload2)) % 4
        if pad:
            payload2 += "=" * pad
        raw = base64.b64decode(payload2, validate=False)
    return raw, mime


# ===================== EUIPO Auth =====================
_euipo_sandbox_access_token: Optional[str] = None
_euipo_sandbox_token_expires_at: float = 0.0

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
        r = requests.post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials", "client_id": cid, "client_secret": csec, "scope": "uid"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        _euipo_sandbox_access_token = data.get("access_token")
        _euipo_sandbox_token_expires_at = time.time() + data.get("expires_in", 3600) - 60
        print("--- [TOOL LOG] Token OK ---")
        return _euipo_sandbox_access_token
    except RequestException as e:
        print(f"--- [TOOL ERROR] Token error: {e}")
        return None


# ===================== Image normalize =====================
def _looks_like_svg(raw: bytes) -> bool:
    head = raw[:256].lstrip().lower()
    return head.startswith(b"<?xml") or b"<svg" in head

def _to_jpeg_b64_smart(raw_bytes: bytes, content_type: Optional[str] = None, source_hint: str = "") -> Optional[str]:
    """
    Chuẩn hoá bytes -> JPEG base64 (max 512px).
    - HTML/JSON -> bỏ
    - SVG -> PNG (cairosvg nếu có)
    - GIF động -> frame 0
    - EXIF transpose, ép RGB
    """
    try:
        ct = (content_type or "").lower()

        # Nhận diện nhanh các định dạng không phải ảnh mà người dùng hay lỡ chọn
        head = raw_bytes[:12]
        if head.startswith(b"%PDF"):
            print(f"[WARN] User provided a PDF, not an image | src={source_hint}")
            return None
        if head.startswith(b"PK\x03\x04"):
            print(f"[WARN] ZIP/Office file uploaded | src={source_hint}")
            return None

        # SVG?
        if "image/svg+xml" in ct or _looks_like_svg(raw_bytes):
            try:
                import cairosvg
                raw_bytes = cairosvg.svg2png(bytestring=raw_bytes, output_width=512, output_height=512)
                ct = "image/png"
            except Exception as e:
                print(f"[WARN] SVG->PNG failed: {e} | src={source_hint}")
                return None

        # Mở bằng Pillow
        bio = io.BytesIO(raw_bytes)
        try:
            img = Image.open(bio); img.load()
        except Exception as e:
            # log thêm vài byte đầu để soi
            print(f"[WARN] Pillow open failed ({ct}) from {source_hint}: {e} | magic={head.hex()}")
            return None

        # Fix orientation + convert RGB
        try:
            if getattr(img, "is_animated", False):
                img.seek(0)
        except Exception:
            pass
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        img.thumbnail((512, 512))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception as e:
        print(f"[WARN] Normalize img failed (smart): {e} | src={source_hint}")
        return None

def _download_bytes(url: str, timeout: int = 15) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        r = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"Accept": "image/*,application/octet-stream;q=0.8,*/*;q=0.5"},
        )
        r.raise_for_status()
        return r.content, r.headers.get("Content-Type")
    except Exception as e:
        print(f"[WARN] Download failed: {e} | url={url}")
        return None, None


# ===================== EUIPO image endpoints =====================
def _fetch_image_from_endpoints(app_no: str, headers: Dict[str, str], prefer_thumb: bool = True) -> Optional[str]:
    """
    Thử lấy ảnh qua endpoint ảnh khi detail không nhúng inline:
      1) /trademarks/{app}/image/thumbnail (nhanh, nhỏ)
      2) /trademarks/{app}/image          (đầy đủ)
    Trả về JPEG base64 hoặc None.
    """
    base = "https://api-sandbox.euipo.europa.eu/trademark-search/trademarks"
    order = [f"{base}/{app_no}/image/thumbnail", f"{base}/{app_no}/image"]
    if not prefer_thumb:
        order = [order[1], order[0]]

    for url in order:
        try:
            h = dict(headers)
            h["Accept"] = "image/*"
            r = requests.get(url, headers=h, timeout=20, allow_redirects=True)
            if r.status_code != 200:
                print(f"[DEBUG] image endpoint {url} -> {r.status_code}")
                continue
            ct = r.headers.get("Content-Type", "")
            if not ct.startswith("image/") and ct not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
                print(f"[WARN] image endpoint Content-Type not image: {ct} | {url}")
            b64 = _to_jpeg_b64_smart(r.content, ct, source_hint=f"{app_no}:{url.rsplit('/',1)[-1]}")
            if b64:
                print(f"[DEBUG] fetched image via endpoint: {url} | bytes={len(r.content)}")
                return b64
        except Exception as e:
            print(f"[WARN] image endpoint failed: {e} | {url}")
    return None


# ===================== markImage parsing =====================
def _extract_b64_from_obj(obj: Any, app_no: str) -> Optional[str]:
    """Tìm base64/URL ảnh trong 1 dict (nhiều tên field khác nhau)."""
    if not isinstance(obj, dict):
        return None
    # base64 keys
    for k in ("content", "base64", "imageBase64", "imageContent", "data", "binary", "payload"):
        s = obj.get(k)
        if isinstance(s, str) and len(s) > 100:
            try:
                raw, mime_from_prefix = decode_any_base64(s)
                ct = obj.get("contentType") or obj.get("mimeType") or mime_from_prefix
                if raw:
                    return _to_jpeg_b64_smart(raw, ct, source_hint=f"{app_no}:{k}")
            except Exception as e:
                print(f"[WARN] b64 decode failed: {e} | {app_no}:{k}")
    # URL keys
    for k in ("url", "uri", "href", "imageUrl", "imageUri", "markImageUri"):
        u = obj.get(k)
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            raw, ct = _download_bytes(u)
            if raw:
                return _to_jpeg_b64_smart(raw, ct, source_hint=f"{app_no}:{k}")
    return None

def extract_logo_b64_from_detail(app_no: str, headers: Dict[str, str]) -> Optional[str]:
    """
    Gọi /trademarks/{applicationNumber} rồi quét mọi nhánh có thể chứa ảnh (đệ quy).
    Nếu không có inline image, fallback gọi endpoint ảnh /image/thumbnail rồi /image.
    """
    base = "https://api-sandbox.euipo.europa.eu/trademark-search/trademarks"

    def _try_inline(params: Optional[Dict[str, str]]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            r = requests.get(f"{base}/{app_no}", headers=headers, params=params, timeout=20)
            r.raise_for_status()
            detail = r.json()
        except Exception as e:
            print(f"--- [TOOL WARN] Detail fetch failed {app_no}: {e}")
            return None, None

        mark_feature = (detail.get("markFeature") or "UNKNOWN").upper()
        print(f"[DEBUG] detail {app_no} markFeature={mark_feature} keys={list(detail.keys())[:8]}")

        # debug markImage keys nếu có
        mi = detail.get("markImage")
        if isinstance(mi, dict):
            print(f"[DEBUG] markImage keys for {app_no}: {list(mi.keys())}")
            for k in ("content","contentType","imageUrl","imageId","binaryObjectId"):
                if k in mi:
                    v = mi[k]
                    vs = (v[:60] + "...") if isinstance(v, str) and len(v) > 60 else v
                    print(f"[DEBUG] markImage.{k} = {vs}")

        # 1) các đường tắt
        for p in ("markImage", "representation", "figurativeMark", "figurativeRepresentation", "graphicalRepresentation"):
            node = detail.get(p)
            if isinstance(node, dict):
                if isinstance(node.get("image"), dict):
                    b64 = _extract_b64_from_obj(node["image"], app_no)
                    if b64: return b64, detail
                b64 = _extract_b64_from_obj(node, app_no)
                if b64: return b64, detail

        # 2) các danh sách
        for p in ("markImageList", "images", "imageList", "representations", "reproductions",
                  "figurativeReproductions", "graphicalRepresentations"):
            arr = detail.get(p)
            if isinstance(arr, list):
                for it in arr:
                    if isinstance(it, dict) and isinstance(it.get("image"), dict):
                        b64 = _extract_b64_from_obj(it["image"], app_no)
                        if b64: return b64, detail
                    elif isinstance(it, dict):
                        b64 = _extract_b64_from_obj(it, app_no)
                        if b64: return b64, detail

        # 3) đệ quy toàn bộ cây (giới hạn nodes)
        from collections import deque
        q = deque([detail]); seen = 0; max_nodes = 5000
        while q and seen < max_nodes:
            cur = q.popleft(); seen += 1
            if isinstance(cur, dict):
                b64 = _extract_b64_from_obj(cur, app_no)
                if b64: return b64, detail
                for v in cur.values():
                    if isinstance(v, (dict, list)): q.append(v)
            elif isinstance(cur, list):
                for v in cur:
                    if isinstance(v, (dict, list)): q.append(v)

        print(f"[DEBUG] no-markImage-after-recursive app {app_no}")
        return None, detail

    # Lần 1: xin rõ trường con (content,contentType,...)
    fields = (
        "applicationNumber,markFeature,markBasis,"
        "markImage(content,contentType,imageUrl,imageId,binaryObjectId,viennaClasses,imageFormat),"
        "representation(image(content,contentType,imageUrl,imageId,binaryObjectId)),"
        "representations(image(content,contentType,imageUrl,imageId,binaryObjectId)),"
        "reproductions(image(content,contentType,imageUrl,imageId,binaryObjectId)),"
        "figurativeReproductions(image(content,contentType,imageUrl,imageId,binaryObjectId)),"
        "graphicalRepresentations(image(content,contentType,imageUrl,imageId,binaryObjectId))"
    )
    b64, detail1 = _try_inline({"fields": fields})
    if b64:
        return b64

    # Lần 2: bỏ fields (một số record chỉ nhúng content nếu không lọc)
    b64, detail2 = _try_inline(None)
    if b64:
        return b64

    # Lần 3: dùng endpoint ảnh (đã bật)
    print(f"[DEBUG] try image endpoints for app {app_no}")
    b64 = _fetch_image_from_endpoints(app_no, headers, prefer_thumb=True)
    if b64:
        return b64

    print(f"[DEBUG] no inline/endpoint image for {app_no}")
    return None


# ===================== Misc =====================
def _sanitize_for_rsql(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    return re.sub(r"\s+", " ", s).strip()

def _name_score_pair(a: str, b: str) -> float:
    s1 = compare_text_similarity_tool.invoke({"text1": a, "text2": b})
    s2 = compare_text_similarity_tool.invoke({"text1": _sanitize_for_rsql(a), "text2": _sanitize_for_rsql(b)})
    return max(s1, s2)


# ===================== Tools =====================
class TrademarkSearchInput(BaseModel):
    name: str = Field(description="Tên nhãn hiệu cần tra cứu.")
    nice_class: Optional[int] = Field(default=None)
    threshold: Optional[float] = Field(default=None)
    user_logo_b64: Optional[str] = Field(default=None, description="(Optional) Base64 JPEG/PNG logo của người dùng.")

@tool(args_schema=TrademarkSearchInput)
def trademark_search_tool(
    name: str,
    nice_class: Optional[int] = None,
    threshold: Optional[float] = 0.85,
    user_logo_b64: Optional[str] = None,
) -> List[Dict[str, Any]]:
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
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-IBM-Client-Id": os.environ.get("EU_SANDBOX_ID"),
    }

    api = "https://api-sandbox.euipo.europa.eu/trademark-search/trademarks"
    # --- Build query gốc ---
    q_base = f"wordMarkSpecification.verbalElement==*{sanitized_name}*"
    if nice_class:
        q_base += f" and niceClasses=={nice_class}"
    # cố tăng xác suất có ảnh khi có logo người dùng
    params = {"size": 75}

    def _fetch_list(q: str) -> List[Dict[str, Any]]:
        try:
            print(f"[DEBUG] list query: {q}")
            r = requests.get(api, headers=headers, params={**params, "query": q}, timeout=20)
            r.raise_for_status()
            js = r.json() or {}
            items = js.get("trademarks", [])
            print(f"[DEBUG] list size: {len(items)}")
            return items
        except (JSONDecodeError, RequestException) as e:
            print(f"[TOOL WARN] list API error: {e} | q={q}")
            return []

    # --- Lấy/chuẩn hoá logo người dùng từ arg hoặc CTX ---
    picked_src = None
    for cand in (user_logo_b64, USER_LOGO_B64_CTX):
        if not cand:
            continue
        if isinstance(cand, str) and cand.strip().lower().startswith("logo_b64"):
            continue
        raw, _ = decode_any_base64(cand)
        if raw and len(raw) >= 64:
            picked_src = cand
            break
    print("[DEBUG] _src type:", type(picked_src).__name__)
    print("[DEBUG] _src prefix:", (str(picked_src)[:80] if picked_src else None))

    user_b64 = None
    if picked_src:
        try:
            raw, _ = decode_any_base64(picked_src)
            print("[DEBUG] user_logo len:", len(raw) if raw else 0,
                  "head:", (raw[:8].hex() if raw else ""))
            if raw and len(raw) >= 64:
                user_b64 = _to_jpeg_b64_smart(raw, None, source_hint="user_logo")
            else:
                print(f"[WARN] User logo too short: {0 if not raw else len(raw)} bytes")
        except Exception as e:
            print(f"--- [TOOL WARN] User logo invalid base64: {e}")

    has_user_logo = bool(user_b64)
    print(f"[DEBUG] has_user_logo={has_user_logo}")

    # --- Tách luồng: WORD (tên) và NON-WORD (logo+Tên khi có logo) ---
    if has_user_logo:
        q_word = q_base + " and markFeature==WORD"
        # Ưu tiên EU_TRADEMARK để tăng cơ hội có ảnh inline/endpoint
        q_fig  = q_base + " and markFeature!=WORD and markBasis==EU_TRADEMARK"
        candidates_word = _fetch_list(q_word)
        candidates_fig  = _fetch_list(q_fig)
    else:
        candidates_word = _fetch_list(q_base)
        candidates_fig  = []

    # Thống kê markFeature (nếu server có trả ở list)
    try:
        from collections import Counter
        mf = Counter([(c.get("markFeature") or "UNKNOWN") for c in (candidates_word + candidates_fig)])
        print("[DEBUG] markFeature dist:", dict(mf))
    except Exception:
        pass

    thr = threshold if threshold is not None else 0.85
    filtered: List[Dict[str, Any]] = []

    # --- 1) WORD: chỉ điểm tên ---
    for c in candidates_word:
        cand_name = (c.get("wordMarkSpecification") or {}).get("verbalElement", "")
        if not cand_name:
            continue
        name_score = _name_score_pair(original_name, cand_name)
        print(f"[SCORE WORD] '{original_name}' vs '{cand_name}': {name_score:.3f}")
        if name_score >= thr:
            c["similarity_score"] = round(float(name_score), 3)
            c["logo_similarity"] = None
            c["combined_score"]  = float(c["similarity_score"])
            filtered.append(c)

    # --- 2) NON-WORD: tên + (nếu có) logo ---
    want_images, got, misses = 5, 0, 0
    max_misses = 20  # dừng sớm nếu sandbox không có ảnh cho nhiều bản ghi

    for c in candidates_fig:
        cand_name = (c.get("wordMarkSpecification") or {}).get("verbalElement", "")
        if not cand_name:
            continue
        name_score = _name_score_pair(original_name, cand_name)
        print(f"[SCORE FIG] '{original_name}' vs '{cand_name}': {name_score:.3f}")
        if name_score < thr:
            continue
        c["similarity_score"] = round(float(name_score), 3)
        c["logo_similarity"]  = None

        if has_user_logo:
            app_no = str(c.get("applicationNumber"))
            # 1) inline / 2) không fields / 3) endpoint ảnh
            cand_b64 = extract_logo_b64_from_detail(app_no, headers)
            if not cand_b64:
                misses += 1
                print(f"[DEBUG] figurative but no image: app={app_no}, misses={misses}")
                if misses >= max_misses:
                    print("[DEBUG] too many misses; skip logo compare for rest")
                    # vẫn cộng vào filtered với điểm tên
                else:
                    # tiếp tục duyệt các record khác
                    pass
            else:
                print(f"[DEBUG] image ready for app {app_no}, b64len={len(cand_b64)}")
                try:
                    ls = compare_logo_similarity_tool.invoke(
                        {"user_logo_b64": user_b64, "candidate_logo_b64": cand_b64}
                    )
                    print(f"[DEBUG] compare result for app {app_no}: {ls}")
                    if ls is not None:
                        c["logo_similarity"] = float(ls)
                        print(f"--- [CLIP LOG] {app_no} logo_sim={c['logo_similarity']}")
                        got += 1
                except Exception as e:
                    print(f"--- [TOOL WARN] CLIP compare failed: {e}")

                if got >= want_images:
                    print(f"[DEBUG] reached target images: {got}")
                    # không break; vẫn để tên các record khác vào filtered

        if c["logo_similarity"] is not None:
            c["combined_score"] = round(0.5 * c["similarity_score"] + 0.5 * c["logo_similarity"], 4)
        else:
            c["combined_score"] = float(c["similarity_score"])

        filtered.append(c)

    if not filtered:
        return [{"message": f"Không ứng viên đạt ngưỡng >= {thr}"}]

    filtered.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    top5 = filtered[:5]

    # Trả về gọn: không bao gồm ảnh
    out: List[Dict[str, Any]] = []
    for c in top5:
        out.append(
            {
                "applicationNumber": c.get("applicationNumber"),
                "verbalElement": (c.get("wordMarkSpecification") or {}).get("verbalElement"),
                "niceClasses": c.get("niceClasses", []),
                "name_similarity": float(c.get("similarity_score", 0.0)),
                "logo_similarity": (None if c.get("logo_similarity") is None else float(c["logo_similarity"])),
                "combined_score": float(c.get("combined_score", 0.0)),
            }
        )

    # Nếu không record nào có ảnh → thêm ghi chú UX
    if has_user_logo and all(x.get("logo_similarity") is None for x in out):
        out[0]["note"] = "Sandbox/record không cung cấp ảnh; hệ thống chỉ tính điểm tên."

    print(f"--- [SEARCH LOG] Done. Top-{len(out)} ---")
    return out