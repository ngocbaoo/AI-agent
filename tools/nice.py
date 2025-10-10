import json
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os

# --- LLM Setup (Tái sử dụng các biến môi trường) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY", "ollama")

# Sử dụng một LLM riêng cho việc phân loại
classifier_llm = ChatOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    model=OLLAMA_MODEL,
    temperature=0,
)

# --- Tool mới ---
@tool
def suggest_nice_class_tool(product_description: str) -> str:
    """
    Dựa vào mô tả sản phẩm, suy luận ra các Nhóm Nice phù hợp nhất.
    Trả về một chuỗi chứa các số của nhóm, cách nhau bởi dấu phẩy (ví dụ: '9, 39')..
    """
    print(f"--- [TOOL LOG] Bắt đầu suy luận Nhóm Nice cho mô tả: '{product_description[:50]}...' ---")
    
    # 1. Tải dữ liệu từ file JSON
    try:
        # Đảm bảo file nice_classes.json nằm cùng cấp với thư mục chạy app.py
        with open('data/nice_classes.json', 'r', encoding='utf-8') as f:
            nice_data = json.load(f)
        
        # Chuyển dữ liệu thành một chuỗi dễ đọc cho LLM
        # File JSON của bạn là một list of dicts, cần điều chỉnh cách đọc
        nice_list_str = "\n".join([f"- Nhóm {item['class']}: {item['description']}" for item in nice_data])
    except Exception as e:
        print(f"--- [TOOL ERROR] Không thể đọc file nice_classes.json: {e}")
        return "Lỗi: Không tìm thấy hoặc không thể đọc file phân loại Nice."

    # 2. Tạo prompt để yêu cầu LLM phân loại
    classification_prompt = ChatPromptTemplate.from_template(
        """Bạn là chuyên gia phân loại sản phẩm.
        Nhiệm vụ của bạn là đọc mô tả sản phẩm và chọn ra **TẤT CẢ CÁC Nhóm Nice có thể phù hợp** từ danh sách dưới đây.
        Hãy suy nghĩ kỹ, một sản phẩm có thể thuộc nhiều nhóm.
        Chỉ trả lời bằng **các số của nhóm, cách nhau bởi dấu phẩy**, không giải thích gì thêm. Ví dụ: 9, 42

        **Mô tả sản phẩm:**
        {description}

        **Danh sách các Nhóm Nice:**
        {nice_list}

        **Các Nhóm Nice phù hợp là (chỉ ghi số, cách nhau bởi dấu phẩy):**
        """
    )
    
    # 3. Tạo và thực thi chain
    chain = classification_prompt | classifier_llm | StrOutputParser()
    
    result = chain.invoke({
        "description": product_description,
        "nice_list": nice_list_str
    })
    
    # Lấy số từ kết quả trả về của LLM
    final_class = "".join(filter(str.isdigit, result))
    print(f"--- [TOOL LOG] LLM đã suy luận ra Nhóm Nice: {final_class} ---")
    
    if not final_class:
        print("--- [TOOL WARN] LLM không trả về số, trả về None ---")
        return None
        
    return final_class