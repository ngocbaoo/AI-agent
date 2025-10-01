from langchain_core.tools import tool
from typing import List, Dict
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
from langchain_community.utilities import GoogleSearchAPIWrapper

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


# --- Định nghĩa Tools ---
# --- Nhóm 1: Công cụ Tra cứu API (API Search Tools) ---

@tool
def trademark_search_tool(name: str, market: str, nice_class: int) -> List[Dict]:
    """
    Tra cứu tên nhãn hiệu trên một thị trường cụ thể theo phân loại Nice.
    Ví dụ thị trường: 'VN', 'EU', 'US', 'WIPO'.
    """
    print(f"--- [TOOL LOG] Giả lập tra cứu nhãn hiệu '{name}' tại '{market}', nhóm Nice {nice_class} ---")
    # Giả lập kết quả trả về từ API, có cấu trúc rõ ràng
    return [
        {"name": f"SimilarBrand_{name}_1", "owner": "ABC Corp", "status": "registered"},
        {"name": f"Another_{name}_Brand", "owner": "XYZ Ltd", "status": "pending"}
    ]

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

# --- Nhóm 2: Công cụ So khớp Cục bộ (Local Matching Tools) ---

@tool
def compare_text_similarity_tool(text1: str, text2: str) -> float:
    """
    Đo lường mức độ tương đồng giữa hai chuỗi văn bản (0.0 đến 1.0).
    """
    print(f"--- [TOOL LOG] Giả lập so sánh văn bản: '{text1}' vs '{text2}' ---")
    # Giả lập một điểm số tương đồng
    import random
    return round(random.uniform(0.7, 0.95), 2)

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
    trademark_search_tool,
    design_search_tool,
    patent_search_tool,
    compare_text_similarity_tool,
    compare_logo_similarity_tool,
    legal_rag_tool
]