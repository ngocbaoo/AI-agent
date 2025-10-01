import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_and_enrich_docs(directory_path: str, document_map: dict) -> list:
    all_docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            doc_number = document_map.get(filename, filename)
            loader = PyPDFLoader(os.path.join(directory_path, filename))
            docs = loader.load()
            
            # Add metadata for every page
            for doc in docs:
                doc.metadata['document_number'] = doc_number
            
            all_docs.extend(docs)
    return all_docs 


# Split into chunks
def split_text_into_chunks(all_docs: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    
    split_doc = text_splitter.split_documents(all_docs)
    return split_doc


# Store into a ChromaDB
def create_and_persist_db(chunk: list, persist_directory: str):
    model_name = 'bkai-foundation-models/vietnamese-bi-encoder'
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )
    persist_directory = "./vector_db"
    
    Chroma.from_documents(
        documents=chunk,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Hoàn tất! Đã lưu thành công Vector DB vào thư mục '{persist_directory}'")

    

if __name__ == "__main__":
    # Directory path and document map
    directory_path = "D:\\Work\\AI agent\\src"
    document_map = {
    "01_2008_TT-BKHCN_62793.pdf": "Thông tư số 01/2008/TT-BKHCN",
    "01_2012_ND-CP_133717.pdf": "Nghị định số 01/2012/NĐ-CP",
    "08_2023_TT-BVHTTDL_568340.pdf": "Thông tư số 08/2023/TT-BVHTTDL",
    "11_VBHN-VPQH_556862.pdf": "Luật Sở hữu trí tuệ số 11/VBHN-VPQH",
    "17_2023_ND-CP_565147.pdf": "Nghị định số 17/2023/NĐ-CP",
    "23_2023_TT-BKHCN_589710.pdf": "Thông tư số 23/2023/TT-BKHCN",
    "50_2005_QH11_7022.pdf":"Luật Sở hữu trí tuệ số 50/2005/QH11",
    "65_2023_ND-CP_576846.pdf":"Nghị định số 65/2023/NĐ-CP",
    "79_2023_ND-CP_586871.pdf":"Nghị định số 79/2023/NĐ-CP",
    "99_2013_ND-CP_205677.pdf":"Nghị định số 99/2013/NĐ-CP",
    "105_2006_ND-CP_14289.pdf":"Nghị định số 105/2006/NĐ-CP",
    "119_2010_ND-CP_116778.pdf":"Nghị định số 119/2010/NĐ-CP",
    "126_2021_ND-CP_499367.pdf":"Nghị định số 126/2021/NĐ-CP",
    "131_2013_ND-CP_210029.pdf":"Nghị định số 131/2013/NĐ-CP",
    "154_2018_ND-CP_399619.pdf":"Nghị định số 154/2018/NĐ-CP",
    "VanBanGoc_11.2015.TT.BKHCN.pdf":"Thông tư số 11/2015/TT-BKHCN",
    "VanBanGoc_14-2016-TTLT-BTTTT-BKHCN.pdf":"Thông tư liên tịch số 14/2016/TTLT-BTTTT-BKHCN",
    "VanBanGoc_Thong tu 05_2016_TTBKHCN_BKHĐT.pdf":"Thông tư liên tịch số 05/2016/TTLT-BKHCN-BKHĐT",
    "VanBanGoc_13.2015.TT.BTC.pdf":"Thông tư số 13/2015/TT-BTC",
    "VanBanGoc_211-2016-TT-BTC.pdf":"Thông tư số 211/2016/TT-BTC",
    "VanBanGoc_263-2016-TT-BTC.pdf":"Thông tư số 263/2016/TT-BTC",
    "VanBanGoc_04_2012_TT-BKHCN.pdf":"Thông tư số 04/2012/TT-BKHCN",
    "VanBanGoc_18_2011_TT-BKHCN.pdf":"Thông tư số 18/2011/TT-BKHCN",
    "VanBanGoc_15_2012_TT-BVHTTDL.pdf":"Thông tư số 15/2012/TT-BVHTTDL",
    "VanBanGoc_05.2013.TT.BKHCN.pdf":"Thông tư số 05/2013/TT-BKHCN",
    "VanBanGoc_13_2010_TT-BKHCN.pdf":"Thông tư số 13/2010/TT-BKHCN",
    "114_2013_ND-CP_209071.pdf":"Nghị định số 114/2013/NĐ-CP",
    "98_2011_ND-CP_131014.pdf":"Nghị định số 98/2011/NĐ-CP",
    "88_2010_ND-CP_110399.pdf":"Nghị định số 88/2010/NĐ-CP",
    "122_2010_ND-CP_117020.pdf":"Nghị định số 122/2010/NĐ-CP",
    "103_2006_ND-CP_14288.pdf":"Nghị định số 103/2006/NĐ-CP",
    "22_2018_ND-CP_351778.pdf":"Nghị định số 22/2018/NĐ-CP",
    "Ngh_ d_nh s_ 28.2017.ND-CP.pdf":"Nghị định số 28/2017/NĐ-CP",
    "01.2007.TT.BKHCN.pdf":"Thông tư số 01/2007/TT-BKHCN",
    "16.2013.TT.BNNPTNT.pdf":"Thông tư số 16/2013/TT-BNNPTNT",
    "khongso_12722.pdf":"Hiệp định về các khía cạnh liên quan tới thương mại của quyền sở hữu trí tuệ",
    "Khongso_60106.pdf":"Công ước Berne về bảo hộ các tác phẩm văn học nghệ thuật",
    "Khongso_66772.pdf":"Công ước quốc tế về bảo hộ người biểu diễn, nhà xuất bản ghi âm, tổ chức phát sóng",
    "Khongso_62697.pdf":"Công ước Paris về bảo hộ sở hữu công nghiệp",
    "Khongso_62652.pdf":"Công ước quốc tế về bảo hộ giống cây trồng mới",
    "Khongso_66846.pdf":"Hiệp ước WIPO về quyền tác giả (WCT) 1996",
    "Khongso_66838.pdf":"Hiệp ước WIPO về biểu diễn và bản ghi âm (WPPT) 1996",
    "Khong_so_10186.pdf":"Thỏa ước Madrid về đăng ký quốc tế nhãn hiệu hàng hóa",
    "KhongSo_61742.pdf":"Nghị định thư liên quan đến thỏa ước Madrid về đăng ký quốc tế nhãn hiệu hàng hóa",
    "Khongso_228916.pdf":"Thỏa ước Lahay về đăng ký quốc tế kiểu dáng công nghiệp",
    "HiepDinhCPTPP.pdf":"Hiệp định đối tác toàn diện và tiến bộ xuyên Thái Bình Dương (CPTPP)",
    "Khongso_63171.pdf":"Hiệp định giữa Chính phủ Cộng hòa xã hội chủ nghĩa Việt Nam và Chính phủ Hợp chủng quốc Hoa Kỳ về thiết lập quan hệ quyền tác giả",
    "Khongso_63170.pdf":"Hiệp định giữa Chính phủ Cộng hòa xã hội chủ nghĩa Việt Nam và Chính phủ Liên bang Thụy Sĩ về bảo hộ sở hữu trí tuệ",
    "Khongso_11754.pdf":"Hiệp định giữa Chính phủ Cộng hòa xã hội chủ nghĩa Việt Nam và Chính phủ Hợp chủng quốc Hoa Kỳ về quan hệ thương mại"
    }

    loaded_documents = load_and_enrich_docs(directory_path, document_map)
    #print(loaded_documents[0].metadata)
    
    # split
    splitted_docs = split_text_into_chunks(loaded_documents)
    #print(splitted_docs[0].page_content)
    
    # Store vectors in a folder
    VECTOR_DB_PATH = "./vector_db"
    create_and_persist_db(chunk=splitted_docs, persist_directory=VECTOR_DB_PATH)