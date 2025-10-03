import streamlit as st
from graph import app
from langchain_core.messages import HumanMessage, AIMessage
import base64
import contextlib
import io

def get_image_base64(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        base64_encoded = base64.b64encode(bytes_data).decode()
        return base64_encoded
    return None

# --- SỬA LỖI 1: Cập nhật CSS mạnh mẽ hơn ---
st.markdown("""
<style>
    /* Nhắm đến container markdown chính */
    [data-testid="stMarkdownContainer"] {
        overflow-wrap: break-word;
        word-wrap: break-word;
        white-space: pre-wrap !important; /* Quan trọng: áp dụng cho cả khối code */
    }
    /* Đảm bảo các thẻ con cũng tuân thủ */
    [data-testid="stMarkdownContainer"] * {
        overflow-wrap: break-word;
        word-wrap: break-word;
        white-space: pre-wrap !important;
    }
</style>
""", unsafe_allow_html=True)
# ------------------------------------------

# --- Thiết lập Giao diện và Trạng thái ---
st.set_page_config(page_title="Trợ lý AI Tư vấn SHTT", page_icon="⚖️", layout="wide")
st.title("⚖️ Trợ lý AI Tư vấn Sở hữu trí tuệ")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# --- Giai đoạn 1: Hiển thị Form ---
if not st.session_state.analysis_done:
    st.info("Bước 1: Cung cấp thông tin sản phẩm để nhận phân tích ban đầu.")
    with st.form("product_form"):
        # ... (phần code trong form giữ nguyên) ...
        product_name = st.text_input("Tên sản phẩm/nhãn hiệu dự kiến", "Adios")
        description = st.text_area("Mô tả ngắn về sản phẩm", "Giày thể thao")
        market = st.selectbox("Thị trường mục tiêu", ["EU", "US", "VN", "WIPO"])
        filter_by_nice = st.checkbox("Lọc theo Nhóm Nice", value=True)
        nice_class = st.number_input("Nhóm Nice dự kiến", min_value=1, max_value=45, value=25) if filter_by_nice else None
        uploaded_logo = st.file_uploader("Tải lên logo của bạn (tùy chọn)", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("🚀 Bắt đầu Phân tích")

        if submitted:
            logo_b64_string = get_image_base64(uploaded_logo)
            logo_info = f"- Logo Input: {'Có' if logo_b64_string else 'Không có'}"
            initial_prompt = f"""
            Hãy phân tích rủi ro SHTT cho hồ sơ sau...
            HỒ SƠ:
            - Tên: {product_name}
            - Mô tả: {description}
            - Thị trường: {market}
            - Nhóm Nice: {nice_class if nice_class else 'Không xác định'}
            - {logo_info}
            """
            
            st.session_state.messages.append(HumanMessage(content=initial_prompt))

            with st.chat_message("ai"):
                log_stream = io.StringIO() 
                with st.spinner("Agent đang phân tích, vui lòng chờ..."):
                    with contextlib.redirect_stdout(log_stream): 
                        # --- SỬA LỖI 2: Thêm một dòng log khởi đầu ---
                        print("--- Bắt đầu phiên làm việc của Agent ---")
                        # ------------------------------------------
                        stream = app.stream({"messages": st.session_state.messages})
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in stream:
                            node_name = list(chunk.keys())[0]
                            if node_name == "agent" and chunk["agent"]["messages"][-1].content:
                                content = chunk["agent"]["messages"][-1].content
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                
                log_output = log_stream.getvalue()
                if log_output:
                    with st.expander("📄 Xem Log hoạt động của Agent"):
                        st.text_area("Logs", log_output, height=300)

                if full_response:
                    st.session_state.messages.append(AIMessage(content=full_response))
                    st.session_state.analysis_done = True
                    st.rerun()
                else:
                    st.error("Agent không thể hoàn thành phân tích. Vui lòng kiểm tra Log.")
            
# --- 3. Giai đoạn 2: Giao diện Chat ---
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
                # --- THAY ĐỔI 2 (Lần 2): Bỏ phần hiển thị "thinking" ---
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