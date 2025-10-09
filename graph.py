import operator
import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List, Dict, Any, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
import warnings
warnings.filterwarnings('ignore')

from tools import tools

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY", "ollama")

llm = ChatOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    model=OLLAMA_MODEL,
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    """
    Nhận toàn bộ lịch sử hội thoại, áp dụng một bộ quy tắc,
    và quyết định hành động tiếp theo.
    """
    print("--- AGENT: Đang quyết định hành động tiếp theo... ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """Bạn là trợ lý AI SHTT.

        NHIỆM VỤ:
        - Phân tích hồ sơ, gọi công cụ tra cứu phù hợp, và xuất báo cáo chiến lược + bảng Top-5 dấu hiệu gần giống nhất.

        QUY TRÌNH BẮT BUỘC
        1) Đọc mục HỒ SƠ trong tin nhắn của người dùng:
            - Nếu có 'Logo_b64_present: yes' thì coi 'Logo_b64:' là base64 logo người dùng.
        2) Tra cứu nhãn hiệu bằng công cụ:
            - Gọi `trademark_search_tool` với:
            • name = tên trong hồ sơ
            • nice_class = nếu có
            • threshold = 0.8
            • user_logo_b64 = Logo_b64 (nếu present)
        3) Dựa vào kết quả để đưa ra kết luận rủi ro + khuyến nghị.

        QUY TẮC
        - BẮT BUỘC phải dùng tool để lấy dữ liệu, không bịa, không trả lời linh tinh.
        - Trả lời bằng tiếng việt
        - Tra cứu luật bắt buộc phải dùng legal_rag_tool
        - Khi trích dẫn luật: nêu rõ Điều X và số hiệu văn bản.
        - KHÔNG nhắc đến tên công cụ hay con số threshold trong phần trả lời.
        - ĐƯỢC phép hiển thị điểm tương đồng 0..1 trong bảng Top-5.
        - BẮT BUỘC phải trả kết quả top 5 về dạng bảng không được trả về dạng json

        ĐỊNH DẠNG CÂU TRẢ LỜI CUỐI
        1. Tóm tắt rủi ro & căn cứ pháp lý (có trích dẫn).
        2. Khuyến nghị chiến lược.
        3. Bảng Top-5 gần giống nhất (nếu có), cột:
            - Tên
            - Điểm tương đồng tên (similarity_score)
            - Điểm tương đồng logo (logo_similarity, nếu có)
            - Ảnh (nếu có markImageBase64, hiển thị; nếu không có thì ghi 'N/A')
     """),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Tạo một chain mới kết hợp prompt và llm (đã được bind tools)
    agent_chain = prompt | llm_with_tools

    # Gọi chain với toàn bộ state['messages']
    response = agent_chain.invoke({"messages": state['messages']})
    
    # Trả về một AIMessage (có thể chứa tool_call hoặc không)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    # Nếu tin nhắn cuối cùng có yêu cầu gọi tool
    if last_message.tool_calls:
        return "continue"
    # Nếu không, kết thúc và trả lời người dùng
    else:
        return "end"
    
# Khởi tạo đồ thị mới    
workflow = StateGraph(AgentState)

# Chỉ cần 2 node: "agent" để suy nghĩ và "executor" để hành động
workflow.add_node("agent", agent_node)
workflow.add_node("executor", ToolNode(tools))

# Đặt điểm bắt đầu
workflow.set_entry_point("agent")

# Thêm cạnh có điều kiện
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "executor",
        "end": END
    }
)

# Sau khi thực thi tool, luôn quay lại agent để nó quyết định bước tiếp theo
workflow.add_edge("executor", "agent")

# Biên dịch agent
app = workflow.compile()
print("\nAgent đã được biên dịch thành công với kiến trúc mới!")