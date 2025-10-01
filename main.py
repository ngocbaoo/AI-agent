from graph import app
from langchain_core.messages import HumanMessage, AIMessage

def run_agent():
    print("Chào mừng đến với Trợ lý AI Tư vấn Sở hữu trí tuệ!")
    print("="*50)

    # Tạo hồ sơ sản phẩm mẫu
    product_profile = {
        "product_name": "Cà Phê Ban Mê",
        "description": "Một loại cà phê rang xay đậm vị...",
        "market": "VN",
        "nice_class": 30,
    }

    # Tạo câu hỏi ban đầu cho agent
    initial_question = f"""
    - Tên: {product_profile['product_name']}
    - Mô tả: {product_profile['description']}
    - Thị trường: {product_profile['market']}
    - Nhóm Nice: {product_profile['nice_class']}
    
    Đối tượng có được luật sở hữu trí tuệ bảo hộ không (hãy trích dẫn điều luật và số hiệu văn bản).
    """

    # Đầu vào bây giờ chỉ là một danh sách tin nhắn
    inputs = [HumanMessage(content=initial_question)]

    # Stream kết quả để xem quá trình agent làm việc
    for event in app.stream({"messages": inputs}):
        for value in event.values():
            if isinstance(value["messages"][-1], AIMessage):
                # In ra câu trả lời cuối cùng của agent
                if not value["messages"][-1].tool_calls:
                    print("\n✅ PHÂN TÍCH HOÀN TẤT!")
                    print("="*50)
                    print(value["messages"][-1].content)

if __name__ == "__main__":
    run_agent()