import fitz
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def init_page():
    st.set_page_config(
        page_title="Upload PDF(s)",
        page_icon="▤"
    )
    st.sidebar.title("옵션")


def init_messages():
    clear_button = st.sidebar.button("DB 초기화", key="clear")

    if clear_button and "vectorstore" in st.session_state:
        del st.session_state["vectorstore"]
        st.success("벡터 DB가 초기화되었습니다.")


def get_pdf_text():
    pdf_file = st.file_uploader(
        label="PDF를 업로드하세요",
        type="pdf"
    )

    if pdf_file:
        pdf_text = ""

        with st.spinner("PDF 로딩 중..."):
            pdf_doc = fitz.open(
                stream=pdf_file.read(),
                filetype="pdf"
            )

            for page in pdf_doc:
                pdf_text += page.get_text()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small",
            chunk_size=500,
            chunk_overlap=0,
        )

        return text_splitter.split_text(pdf_text)

    return None


def build_vector_store(pdf_text):
    selected_model = st.session_state.get("selected_model")

    # Embedding은 OpenAI 사용 → GPT 모델만 가능
    if selected_model not in ["GPT-5 mini", "GPT-5.1"]:
        st.error("PDF 업로드 및 벡터 저장은 OpenAI API Key가 필요합니다.")
        return

    openai_api_key = st.session_state.get("openai_api_key")

    if not openai_api_key:
        st.sidebar.warning("OpenAI API Key를 입력해주세요.")
        st.experimental_rerun()  # 페이지를 다시 로드하여 API Key 입력 위치로 이동
        return

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

    with st.spinner("벡터 스토어 저장 중..."):
        if "vectorstore" in st.session_state:
            st.session_state.vectorstore.add_texts(pdf_text)
        else:
            st.session_state.vectorstore = FAISS.from_texts(
                pdf_text,
                embeddings
            )

    st.success("벡터 스토어 저장 완료!")


def page_pdf_upload_and_build_vector_db():
    st.title("PDF 업로드 ▤")

    pdf_text = get_pdf_text()

    if pdf_text:
        build_vector_store(pdf_text)


def main():
    init_page()
    init_messages()
    page_pdf_upload_and_build_vector_db()


if __name__ == "__main__":
    main()