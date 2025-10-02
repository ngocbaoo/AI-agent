from langchain_core.tools import tool
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
load_dotenv()

# For RAG
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# For Search
import time
import requests
from requests.exceptions import RequestException, JSONDecodeError
from thefuzz import fuzz
from langchain_core.pydantic_v1 import BaseModel, Field

# --- RAG ---
model_name = "bkai-foundation-models/vietnamese-bi-encoder"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

vector_db_path = "./vector_db"
vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)

gemini_key = os.environ.get("GOOGLE_API_KEY")
retriever = vectorstore.as_retriever()
rag_llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=gemini_key)
def format_docs(docs):
    formatted_context = ""
    for doc in docs:
        doc_num = doc.metadata.get('document_number', doc.metadata.get('source', 'Không rõ nguồn'))
        formatted_context += f"--- Trích dẫn từ: {doc_num} ---\n{doc.page_content}\n\n"
    return formatted_context.strip()

rag_prompt = ChatPromptTemplate.from_template(
    """Bạn là một trợ lý pháp lý chuyên nghiệp, cẩn thận và chính xác.
    Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách súc tích, chỉ dựa vào các đoạn trích dẫn được cung cấp.

    **QUY TẮC TRÍCH DẪN BẮT BUỘC:**
    1.  Khi bạn lấy thông tin từ một đoạn trích, hãy tìm **số Điều** (ví dụ: "Điều 16", "Điều 84") có trong nội dung của đoạn trích đó.
    2.  Kết hợp **số Điều** bạn tìm được với **số hiệu văn bản** được cung cấp trong tiêu đề trích dẫn (ví dụ: "--- Trích dẫn từ: Nghị định số 65/2023/NĐ-CP ---").
    3.  Hãy trích dẫn theo mẫu sau: **(theo Điều X của [Số hiệu văn bản])**.
    4.  **TUYỆT ĐỐI KHÔNG** được đề cập đến tên file, đường dẫn, hay số trang.

    ---
    **CÁC TRÍCH DẪN ĐƯỢC CUNG CẤP:**
    {context}
    ---

    Dựa vào các quy tắc và ngữ cảnh trên, hãy trả lời câu hỏi sau:
    **Câu hỏi:** {question}
    """
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | rag_llm
    | StrOutputParser()
)


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



# --- Định nghĩa Tools ---
# --- Công cụ So khớp Cục bộ (Local Matching Tools) ---

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
    # Giả lập một điểm số tương đồng
    import random
    return round(random.uniform(0.6, 0.9), 2)


# --- Công cụ Tra cứu API (API Search Tools) ---
class EUTMSearchInput(BaseModel):
    name: str = Field(description="Tên nhãn hiệu cần tra cứu")
    nice_class: Optional[int] = Field(default=None, description="Nhóm Nice (Tùy chọn)")
    
@tool(args_schema=EUTMSearchInput)
def euipo_trademark_search_tool(name: str, nice_class: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    (SANDBOX) Tra cứu nhãn hiệu trên môi trường thử nghiệm của EUIPO.
    """
    print(f"--- [TOOL LOG] SANDBOX (RSQL): Bắt đầu tra cứu cho '{name}' ---")

    access_token = _get_euipo_sandbox_access_token()
    if not access_token:
        return [{"error": "Xác thực EUIPO Sandbox (RSQL) thất bại."}]

    api_url = "https://api-sandbox.euipo.europa.eu/trademark-search/trademarks"
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}',
        'X-IBM-Client-Id': os.environ.get("EU_SANDBOX_ID")
    }

    rsql_query = f"wordMarkSpecification.verbalElement==*{name}*"

    params = {
        "query": rsql_query,
        "size": 10, 
        "fields": "applicationNumber,wordMarkSpecification,niceClasses,applicants(name),status" 
    }

    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        search_results = data.get("trademarks", [])
        if not search_results:
            return [{"message": f"(Sandbox RSQL) Không tìm thấy nhãn hiệu nào cho '{name}'."}]
        return search_results
    except (JSONDecodeError, RequestException) as e:
        return [{"error": f"(Sandbox RSQL) Lỗi khi gọi API: {e}"}]
    

class FuzzySearchInput(BaseModel):
    name: str = Field(description="Tên nhãn hiệu cần tra cứu.")
    nice_class: Optional[int] = Field(default=None, description="Nhóm Nice (tùy chọn) để lọc.")
    threshold: float = Field(default=0.9, description="Ngưỡng tương đồng từ 0.0 đến 1.0.")

@tool(args_schema=FuzzySearchInput)
def euipo_fuzzy_trademark_search_tool(name: str, nice_class: Optional[int] = None, threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Thực hiện tìm kiếm gần đúng (fuzzy search) cho một nhãn hiệu.
    Tool này sẽ thực hiện một tìm kiếm rộng trước, sau đó lọc lại
    kết quả dựa trên một ngưỡng tương đồng (threshold).
    """
    print(f"--- [FUZZY SEARCH LOG] Bắt đầu tìm kiếm gần đúng cho '{name}' với ngưỡng {threshold} ---")
    
    broad_search_results = euipo_trademark_search_tool.invoke({"name": name, "nice_class": nice_class})

    if not broad_search_results or "error" in broad_search_results[0] or "message" in broad_search_results[0]:
        print("--- [FUZZY SEARCH LOG] Bước tra cứu mở rộng không có kết quả hoặc bị lỗi. ---")
        return broad_search_results

    print(f"--- [FUZZY SEARCH LOG] Tìm thấy {len(broad_search_results)} ứng cử viên. Bắt đầu lọc... ---")
    
    final_results = []
    for candidate in broad_search_results:
        candidate_name = candidate.get("wordMarkSpecification", {}).get("verbalElement", "")
        if not candidate_name: continue

        similarity_score = compare_text_similarity_tool.invoke({"text1": name, "text2": candidate_name})
        
        if similarity_score >= threshold:
            candidate['similarity_score'] = round(similarity_score, 2)
            final_results.append(candidate)

    if not final_results:
        return [{"message": f"Không tìm thấy nhãn hiệu nào có độ tương đồng >= {threshold} với '{name}'."}]

    final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
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


# --- Nhóm 3: Công cụ Tra cứu Kiến thức (Knowledge Tool) ---

@tool
def legal_rag_tool(query: str) -> str:
    """
    Tra cứu thông tin trong cơ sở dữ liệu văn bản luật SHTT.
    Sử dụng khi cần tìm hiểu về một khái niệm hoặc quy định pháp luật.
    """
    print(f"--- [TOOL LOG] Đang thực thi RAG với câu hỏi: '{query}' ---")
    return rag_chain.invoke(query)


# Gom tất cả các tool vào một danh sách để agent có thể sử dụng
tools = [
    euipo_fuzzy_trademark_search_tool,
    euipo_trademark_search_tool,
    design_search_tool,
    patent_search_tool,
    compare_text_similarity_tool,
    compare_logo_similarity_tool,
    legal_rag_tool
]