import streamlit as st
from graph import app
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Thiết lập Giao diện và Trạng thái ---
st.set_page_config(page_title="Trợ lý AI Tư vấn SHTT", page_icon="⚖️", layout="wide")
st.title("⚖️ Trợ lý AI Tư vấn Sở hữu trí tuệ")

# Khởi tạo session_state để lưu trữ lịch sử chat VÀ trạng thái của app
if "messages" not in st.session_state:
    st.session_state.messages = []
# Thêm một cờ để biết khi nào form đã được submit
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --- 2. Giai đoạn 1: Hiển thị Form khi chưa phân tích ---
if not st.session_state.analysis_done:
    st.info("Bước 1: Cung cấp thông tin sản phẩm để nhận phân tích ban đầu.")
    with st.form("product_form"):
        st.subheader("Hồ sơ Sản phẩm")
        product_name = st.text_input("Tên sản phẩm/nhãn hiệu dự kiến", "Trà Sen An Lạc")
        description = st.text_area("Mô tả ngắn về sản phẩm", "Trà ướp sen Tây Hồ tự nhiên, giúp thư giãn tinh thần.")
        market = st.selectbox("Thị trường mục tiêu", ["VN", "US", "EU", "WIPO"])
        nice_class = st.number_input("Nhóm Nice dự kiến", min_value=1, max_value=45, value=30)
        
        submitted = st.form_submit_button("🚀 Bắt đầu Phân tích")

        if submitted:
            # Tạo mệnh lệnh ban đầu chi tiết
            initial_prompt = f"""
            Hãy phân tích rủi ro sở hữu trí tuệ cho hồ sơ sản phẩm sau đây.
            Bắt đầu bằng việc sử dụng các công cụ tra cứu, sau đó tham khảo luật pháp để đưa ra một báo cáo chiến lược hoàn chỉnh.
            Hãy nhớ rằng tham khảo điều luật nào phải trích dẫn điều khoản số ... và số hiệu văn bản (ví dụ Điều 8 Nghị định số 65/2023/NĐ-CP)

            HỒ SƠ:
            - Tên: {product_name}
            - Mô tả: {description}
            - Thị trường: {market}
            - Nhóm Nice: {nice_class}
            """
            
            # Lưu mệnh lệnh này vào lịch sử chat
            st.session_state.messages.append(HumanMessage(content=initial_prompt))

            # Gọi Agent
            with st.spinner("Agent đang phân tích, vui lòng chờ..."):
                stream = app.stream({"messages": st.session_state.messages})
                
                final_message = None
                for chunk in stream:
                    last_message_state = next(iter(chunk.values()))
                    if last_message_state and "messages" in last_message_state:
                        final_message = last_message_state["messages"][-1]
                
                if final_message and not final_message.tool_calls:
                    # Lưu câu trả lời của AI vào lịch sử
                    st.session_state.messages.append(final_message)
                    # Đánh dấu là đã phân tích xong
                    st.session_state.analysis_done = True
                    # Tải lại trang để chuyển sang giao diện chat
                    st.rerun()
                else:
                    st.error("Agent không thể hoàn thành phân tích. Vui lòng thử lại.")

# --- 3. Giai đoạn 2: Hiển thị Giao diện Chat sau khi đã phân tích ---
if st.session_state.analysis_done:
    st.info("Bước 2: Bạn có thể tiếp tục hỏi các câu hỏi pháp lý liên quan đến báo cáo.")
    
    # Hiển thị toàn bộ lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # Lấy input mới từ người dùng
    if prompt := st.chat_input("Đặt câu hỏi pháp lý..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("human"):
            st.markdown(prompt)

        # Gọi Agent với toàn bộ lịch sử
        with st.chat_message("ai"):
            with st.spinner("Agent đang suy nghĩ..."):
                stream = app.stream({"messages": st.session_state.messages})
                
                final_message = None
                for chunk in stream:
                    last_message_state = next(iter(chunk.values()))
                    if last_message_state and "messages" in last_message_state:
                        final_message = last_message_state["messages"][-1]
                
                if final_message and not final_message.tool_calls:
                    st.markdown(final_message.content)
                    st.session_state.messages.append(final_message)
                else:
                    st.warning("Agent không đưa ra được câu trả lời.")