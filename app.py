# app.py
import streamlit as st
from graph import app
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image
import io, base64

st.set_page_config(page_title="Trợ lý AI Tư vấn SHTT", page_icon="⚖️", layout="wide")
st.title("⚖️ Trợ lý AI Tư vấn Sở hữu trí tuệ")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

def _compress_to_b64(file) -> str:
    img = Image.open(file).convert("RGB")
    img.thumbnail((512, 512))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    buf.close()
    return b64

if not st.session_state.analysis_done:
    st.info("Bước 1: Cung cấp thông tin sản phẩm để nhận phân tích ban đầu.")
    with st.form("product_form"):
        st.subheader("Hồ sơ Sản phẩm")
        product_name = st.text_input("Tên sản phẩm/nhãn hiệu dự kiến", "Panasonic")
        description = st.text_area("Mô tả ngắn về sản phẩm", "")
        market = st.selectbox("Thị trường mục tiêu", ["EU", "US", "VN", "WIPO"])
        filter_by_nice = st.checkbox("Lọc theo Nhóm Nice", value=False)
        nice_class = st.number_input("Nhóm Nice dự kiến", min_value=1, max_value=45, value=5, step=1) if filter_by_nice else None

        # NEW: Uploader logo (in-RAM)
        logo_file = st.file_uploader("Logo (tùy chọn) — PNG/JPG", type=["png", "jpg", "jpeg"])
        user_logo_b64 = None
        if logo_file is not None:
            user_logo_b64 = _compress_to_b64(logo_file)
            st.image(Image.open(io.BytesIO(base64.b64decode(user_logo_b64))), caption="Logo người dùng (preview)", width="stretch")

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
            - Logo_b64_present: {"yes" if user_logo_b64 else "no"}
            - Logo_b64: {user_logo_b64 if user_logo_b64 else ""}
            """
            st.session_state.messages.append(HumanMessage(content=initial_prompt))

            with st.chat_message("ai"):
                with st.spinner("Agent đang phân tích..."):
                    message_placeholder = st.empty()
                    full_response = ""
                    stream = app.stream({"messages": st.session_state.messages})
                    for chunk in stream:
                        node = list(chunk.keys())[0]
                        if node == "agent" and chunk["agent"]["messages"][-1].content:
                            content = chunk["agent"]["messages"][-1].content
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    if full_response:
                        st.session_state.messages.append(AIMessage(content=full_response))
                        st.session_state.analysis_done = True
                        # xoá bản sao b64 khỏi state sau khi phân tích xong (RAM hygiene)
                        user_logo_b64 = None
                    else:
                        st.error("Agent không thể hoàn thành phân tích. Vui lòng kiểm tra lại.")
