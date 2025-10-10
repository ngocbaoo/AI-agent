import streamlit as st
from graph import app
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image, ImageOps
import io, base64
from tools import trademark as search_tools 

st.set_page_config(page_title="Trợ lý AI Tư vấn SHTT", page_icon="⚖️", layout="wide")
st.title("⚖️ Trợ lý AI Tư vấn Sở hữu trí tuệ")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

def file_to_b64jpeg(uploaded_file):
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((512, 512))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")

def _strip_data_url_prefix(s: str) -> str:
    if s and s.startswith("data:"):
        return s.split(",", 1)[1]
    return s

if not st.session_state.analysis_done:
    st.info("Bước 1: Cung cấp thông tin sản phẩm để nhận phân tích ban đầu.")
    with st.form("product_form"):
        st.subheader("Hồ sơ Sản phẩm")
        product_name = st.text_input("Tên sản phẩm/nhãn hiệu dự kiến", "Panasonic")
        description = st.text_area("Mô tả ngắn về sản phẩm", "")
        market = st.selectbox("Thị trường mục tiêu", ["EU", "US", "VN", "WIPO"])

        # NEW: Uploader logo (in-RAM)
        logo_file = st.file_uploader("Logo (tùy chọn) — PNG/JPG", type=["png", "jpg", "jpeg"])
        user_logo_b64 = None
        if logo_file is not None:
            user_logo_b64 = file_to_b64jpeg(logo_file)   
            search_tools.USER_LOGO_B64_CTX = user_logo_b64
            payload = _strip_data_url_prefix(user_logo_b64)
            st.image(
                Image.open(io.BytesIO(base64.b64decode(payload))),
                caption="Logo người dùng (preview)",
                width="stretch",)

        submitted = st.form_submit_button("🚀 Bắt đầu Phân tích")

        if submitted:
            initial_prompt = f"""
            Hãy tra cứu độ tương đồng của hồ sơ sản phẩm sau đây.
            Bắt đầu bằng việc sử dụng các công cụ tra cứu.
            Không được nhắc đến tool sử dụng trong câu trả lời.

            HỒ SƠ:
            - Tên: {product_name}
            - Mô tả: {description}
            - Thị trường: {market}
            - Logo_b64_present: {"yes" if user_logo_b64 else "no"}
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
