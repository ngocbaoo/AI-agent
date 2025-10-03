import os
from dotenv import load_dotenv
from langchain_core.tools import tool

# For RAG
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- RAG Setup ---
model_name = "bkai-foundation-models/vietnamese-bi-encoder"
embedding_model = HuggingFaceEmbeddings(model_name=model_name)
vector_db_path = "./vector_db"
vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=embedding_model)
gemini_key = os.environ.get("GOOGLE_API_KEY")
retriever = vectorstore.as_retriever()
rag_llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=gemini_key)

def format_docs(docs):
    formatted_context = ""
    for doc in docs:
        doc_num = doc.metadata.get('document_number', doc.metadata.get('source', 'Không rõ nguồn'))
        formatted_context += f"--- Trích dẫn từ: {doc_num} ---\n{doc.page_content}\n\n"
    return formatted_context.strip()

rag_prompt = ChatPromptTemplate.from_template(
    """Bạn là một trợ lý pháp lý chuyên nghiệp, cẩn thận và chính xác.
    Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách súc tích, chỉ dựa vào các đoạn trích dẫn được cung cấp.
    **QUY TẮC TRÍCH DẪN BẮT BUỘC:**
    1.  Khi bạn lấy thông tin từ một đoạn trích, hãy tìm **số Điều** (ví dụ: "Điều 16", "Điều 84") có trong nội dung của đoạn trích đó.
    2.  Kết hợp **số Điều** bạn tìm được với **số hiệu văn bản** được cung cấp trong tiêu đề trích dẫn (ví dụ: "--- Trích dẫn từ: Nghị định số 65/2023/NĐ-CP ---").
    3.  Hãy trích dẫn theo mẫu sau: **(theo Điều X của [Số hiệu văn bản])**.
    4.  **TUYỆT ĐỐI KHÔNG** được đề cập đến tên file, đường dẫn, hay số trang.
    ---
    **CÁC TRÍCH DẪN ĐƯỢC CUNG CẤP:**
    {context}
    ---
    Dựa vào các quy tắc và ngữ cảnh trên, hãy trả lời câu hỏi sau:
    **Câu hỏi:** {question}
    """
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | rag_llm
    | StrOutputParser()
)

@tool
def legal_rag_tool(query: str) -> str:
    """
    Tra cứu thông tin trong cơ sở dữ liệu văn bản luật SHTT.
    Sử dụng khi cần tìm hiểu về một khái niệm hoặc quy định pháp luật.
    """
    print(f"--- [TOOL LOG] Đang thực thi RAG với câu hỏi: '{query}' ---")
    return rag_chain.invoke(query)