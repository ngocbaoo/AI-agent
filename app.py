import streamlit as st
from graph import app
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Thiết lập Giao diện và Trạng thái ---
st.set_page_config(page_title="Trợ lý AI Tư vấn SHTT", page_icon="⚖️", layout="wide")
st.title("⚖️ Trợ lý AI Tư vấn Sở hữu trí tuệ")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --- 2. Giai đoạn 1: Hiển thị Form ---
if not st.session_state.analysis_done:
    st.info("Bước 1: Cung cấp thông tin sản phẩm để nhận phân tích ban đầu.")
    with st.form("product_form"):
        st.subheader("Hồ sơ Sản phẩm")
        product_name = st.text_input("Tên sản phẩm/nhãn hiệu dự kiến", "An Lạc")
        description = st.text_area("Mô tả ngắn về sản phẩm", "Sản phẩm chăm sóc sức khỏe từ thảo dược")
        market = st.selectbox("Thị trường mục tiêu", ["EU", "US", "VN", "WIPO"])
        
        filter_by_nice = st.checkbox("Lọc theo Nhóm Nice", value=True)
        
        nice_class = None
        if filter_by_nice:
            nice_class = st.number_input("Nhóm Nice dự kiến", min_value=1, max_value=45, value=5, step=1)
        
        threshold = 0.8
        
        submitted = st.form_submit_button("🚀 Bắt đầu Phân tích")

        if submitted:
            initial_prompt = f"""
            Hãy phân tích rủi ro sở hữu trí tuệ cho hồ sơ sản phẩm sau đây.
            Bắt đầu bằng việc sử dụng các công cụ tra cứu, sau đó tham khảo luật pháp để đưa ra một báo cáo chiến lược hoàn chỉnh.
            Hãy nhớ rằng tham khảo điều luật nào phải trích dẫn điều khoản số ... và số hiệu văn bản (ví dụ Điều 8 Nghị định số 65/2023/NĐ-CP)

            HỒ SƠ:
            - Tên: {product_name}
            - Mô tả: {description}
            - Thị trường: {market}
            - Nhóm Nice: {nice_class if nice_class else 'Không xác định'}
            """
            
            st.session_state.messages.append(HumanMessage(content=initial_prompt))

            with st.chat_message("ai"):
                with st.spinner("Agent đang phân tích, vui lòng chờ..."):
                    # --- Bỏ phần hiển thị "thinking" ---
                    message_placeholder = st.empty()
                    full_response = ""

                    stream = app.stream({"messages": st.session_state.messages})
                    
                    for chunk in stream:
                        node_name = list(chunk.keys())[0]
                        
                        # Chỉ xử lý khi agent trả về nội dung cuối cùng
                        if node_name == "agent" and chunk["agent"]["messages"][-1].content:
                            content = chunk["agent"]["messages"][-1].content
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)
                    
                    if full_response:
                        st.session_state.messages.append(AIMessage(content=full_response))
                        st.session_state.analysis_done = True
                        st.rerun()
                    else:
                        st.error("Agent không thể hoàn thành phân tích. Vui lòng kiểm tra lại.")
            
# --- Giai đoạn 2: Giao diện Chat ---
if st.session_state.analysis_done:
    st.info("Bước 2: Bạn có thể tiếp tục hỏi các câu hỏi pháp lý liên quan đến báo cáo.")
    
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    if prompt := st.chat_input("Đặt câu hỏi pháp lý..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("human"):
            st.markdown(prompt)

        with st.chat_message("ai"):
            with st.spinner("Agent đang suy nghĩ..."):
                # --- Bỏ phần hiển thị "thinking" ---
                message_placeholder = st.empty()
                full_response = ""

                stream = app.stream({"messages": st.session_state.messages})

                for chunk in stream:
                    node_name = list(chunk.keys())[0]
                    
                    if node_name == "agent" and chunk["agent"]["messages"][-1].content:
                        content = chunk["agent"]["messages"][-1].content
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                if full_response:
                    st.session_state.messages.append(AIMessage(content=full_response))