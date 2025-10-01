import operator
import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List, Dict, Any, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
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

api_key = os.environ.get('GOOGLE_API_KEY')
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             google_api_key=api_key).bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    """
    Nhận toàn bộ lịch sử hội thoại, áp dụng một bộ quy tắc,
    và quyết định hành động tiếp theo một cách đáng tin cậy hơn.
    """
    print("--- AGENT: Đang quyết định hành động tiếp theo... ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Bạn là một trợ lý AI chuyên về Sở hữu trí tuệ.
         Nhiệm vụ chính của bạn là phân tích yêu cầu của người dùng và chọn công cụ (tool) phù hợp nhất để thực hiện.

        **QUY TRÌNH SUY LUẬN BẮT BUỘC:**
         1.  **Phân tích yêu cầu:** Đọc kỹ yêu cầu cuối cùng của người dùng.
         2.  **Lựa chọn công cụ:** Dựa trên yêu cầu, hãy chọn MỘT công cụ (tool) phù hợp nhất từ danh sách bạn có để thực hiện bước tiếp theo.
         3.  **Hành động:** Nếu bạn quyết định sử dụng một công cụ, hãy chỉ trả về lời gọi hàm (tool call) đó.
         4.  **Trả lời cuối cùng:** Chỉ sau khi đã thu thập đủ thông tin từ các công cụ, hãy tổng hợp lại và trả lời câu hỏi của người dùng một cách đầy đủ.
         
         **QUY TẮC BỔ SUNG:**
         -   TUYỆT ĐỐI KHÔNG được tự bịa ra lý do không thể dùng tool. Nếu một tool cần thiết, hãy gọi nó.
         -   Nếu câu hỏi liên quan đến luật, PHẢI dùng 'legal_rag_tool'.
         -   Nếu câu hỏi liên quan đến tra cứu nhãn hiệu, PHẢI dùng 'trademark_search_tool'.
         -   Câu trả lời cuối cùng không được nhắc đến tên công cụ.
         """),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Tạo một chain mới kết hợp prompt và llm (đã được bind tools)
    agent_chain = prompt | llm

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