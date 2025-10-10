import json
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
import re

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

@tool
def suggest_nice_class_tool(product_description: str) -> str:
    """
    Dựa vào mô tả sản phẩm, suy luận ra CÁC Nhóm Nice có thể phù hợp.
    Trả về một chuỗi chứa các số của nhóm, cách nhau bởi dấu phẩy (ví dụ: '9, 39').
    """
    print(f"--- [TOOL LOG] Bắt đầu suy luận (nhiều) Nhóm Nice cho mô tả: '{product_description[:50]}...' ---")
    
    try:
        with open(r'data\nice_classes.json', 'r', encoding='utf-8') as f:
            nice_data = json.load(f)
        nice_list_str = "\n".join([f"- Nhóm {item['class']}: {item['description']}" for item in nice_data])
    except Exception as e:
        print(f"--- [TOOL ERROR] Không thể đọc file nice_classes.json: {e}")
        return "Lỗi: Không tìm thấy file phân loại Nice."

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
    
    chain = classification_prompt | classifier_llm | StrOutputParser()
    
    result = chain.invoke({
        "description": product_description,
        "nice_list": nice_list_str
    })
    
    found_numbers = re.findall(r'\d+', result)
    
    valid_numbers = [num for num in found_numbers if 1 <= int(num) <= 45]
    cleaned_result = ", ".join(valid_numbers)
    
    print(f"--- [TOOL LOG] LLM đã suy luận ra các Nhóm Nice: {cleaned_result} ---")
    
    if not cleaned_result:
        print("--- [TOOL WARN] LLM không trả về số nào hợp lệ, trả về None ---")
        return None
        
    return cleaned_result