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
    và quyết định hành động tiếp theo.
    """
    print("--- AGENT: Đang quyết định hành động tiếp theo... ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Bạn là một trợ lý AI chuyên về Sở hữu trí tuệ.
         Nhiệm vụ chính của bạn là phân tích yêu cầu của người dùng và chọn công cụ (tool) phù hợp nhất để thực hiện.

        **QUY TRÌNH SUY LUẬN BẮT BUỘC:**
         1.  **Phân tích yêu cầu:** Đọc kỹ yêu cầu cuối cùng của người dùng.
         2.  **Lựa chọn công cụ:**
             - Nếu yêu cầu liên quan đến tra cứu nhãn hiệu, hãy **ƯU TIÊN** sử dụng công cụ `trademark_search_tool`.
             - **QUAN TRỌNG**: Khi gọi `trademark_search_tool`, hãy **LUÔN LUÔN** truyền vào tham số `threshold=0.8`.
             - Đối với các yêu cầu khác (kiểu dáng, sáng chế, luật), hãy chọn các tool tương ứng.
         3.  **Hành động:** Trả về lời gọi hàm (tool call) cho công cụ đã chọn.
         4.  **Trả lời cuối cùng:** 
             - Chỉ sau khi đã thu thập đủ thông tin từ các công cụ, hãy tổng hợp lại và trả lời câu hỏi của người dùng.
             - Không được hiển thị quá trình suy luận hay nhắc đến việc sử dụng cái gì, ngưỡng bao nhiêu
            
         **QUY TẮC BỔ SUNG:**
         -   TUYỆT ĐỐI KHÔNG được tự bịa ra lý do không thể dùng tool. Nếu một tool cần thiết, hãy gọi nó.
         -   Nếu câu hỏi liên quan đến luật, PHẢI dùng 'legal_rag_tool'.
         -   Nếu câu hỏi liên quan đến tra cứu nhãn hiệu, PHẢI dùng các tool liên quan đến trademark search tùy vào thị trường. 
         -   Câu trả lời cuối cùng không được nhắc đến tên công cụ.
         -   Không được nhắc đến ngưỡng
         -   Khống được nhắc đến điểm tương đồng chỉ nên nhận định rằng chúng có độ giống nhau cao
         
         **QUY TẮC PHÂN TÍCH VÀ NHẬN ĐỊNH BẮT BUỘC:**
        1.  **Diễn giải `similarity_score`**: Khi một công cụ tra cứu trả về kết quả có `similarity_score` cao (ví dụ > 0.8), bạn phải nhận định đây là một **rủi ro xung đột cao**.
        2.  **Nhận diện hành vi "Lách luật"**:
            - **ĐẶC BIỆT**: Nếu tên tìm kiếm ban đầu chứa ký tự đặc biệt hoặc cố tình viết sai (ví dụ: '@dida$', 'guchi') và kết quả có điểm tương đồng cao là một thương hiệu nổi tiếng ('Adidas', 'Gucci'), bạn phải **nhấn mạnh** trong báo cáo rằng đây là một **hành vi cố tình lách luật không hiệu quả** và sẽ bị coi là xâm phạm quyền.
            - Để củng cố nhận định này, hãy giải thích ngắn gọn về khái niệm pháp lý **'khả năng gây nhầm lẫn' (likelihood of confusion)**, dựa trên bài kiểm tra "Thị giác, Âm thanh, và Ý nghĩa".
        3.  **Trích dẫn luật**: Khi đề cập đến một quy định, luôn trích dẫn đầy đủ theo yêu cầu.
        4.  **Cấu trúc báo cáo cuối cùng**: Báo cáo phải rõ ràng, đi thẳng vào kết luận rủi ro và đưa ra đề xuất chiến lược (ví dụ: "Không nên theo đuổi", "Rủi ro cao", "Cần điều chỉnh"...).
        
        **QUY TẮC CHO CÂU TRẢ LỜI CUỐI CÙNG (FORMATTING):**
        -   Câu trả lời cuối cùng phải là một báo cáo phân tích hoàn chỉnh, có cấu trúc rõ ràng và đưa ra đề xuất chiến lược.
        -   **TUYỆT ĐỐI KHÔNG** được hiển thị quá trình suy luận, không nhắc đến tên công cụ đã sử dụng, và không đề cập đến con số `threshold`.
        -   Khi đề cập đến một quy định pháp luật (thông tin lấy từ `legal_rag_tool`), phải trích dẫn đầy đủ theo mẫu: **(theo Điều X của [Số hiệu văn bản])**.
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