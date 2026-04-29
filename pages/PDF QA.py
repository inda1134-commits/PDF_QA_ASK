import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="♬"
    )
    st.sidebar.title("옵션")

def select_model(temperature=0):
    selected_model = st.session_state.get("selected_model")

    if selected_model == "GPT-5 mini":
        api_key = st.session_state.get("openai_api_key")

        if not api_key:
            st.error("OpenAI API Key를 입력해주세요.")
            st.experimental_rerun()

        return ChatOpenAI(
            temperature=temperature,
            model="gpt-5-mini",
            api_key=api_key
        )

    elif selected_model == "GPT-5.1":
        api_key = st.session_state.get("openai_api_key")

        if not api_key:
            st.error("OpenAI API Key를 입력해주세요.")
            st.experimental_rerun()

        return ChatOpenAI(
            temperature=temperature,
            model="gpt-5.1",
            api_key=api_key
        )

    elif selected_model == "Claude Sonnet 4.5":
        api_key = st.session_state.get("anthropic_api_key")

        if not api_key:
            st.error("Anthropic API Key를 입력해주세요.")
            st.experimental_rerun()

        return ChatAnthropic(
            temperature=temperature,
            model="claude-sonnet-4-5-20250929",
            api_key=api_key
        )

    elif selected_model == "Gemini 2.5 Flash":
        api_key = st.session_state.get("google_api_key")

        if not api_key:
            st.error("Google Gemini API Key를 입력해주세요.")
            st.experimental_rerun()

        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model="gemini-2.5-flash",
            google_api_key=api_key
        )

    else:
        st.error("LLM을 먼저 선택해주세요.")
        st.stop()

def init_qa_chain():
    llm = select_model()

    prompt = ChatPromptTemplate.from_template(
        """
다음 배경 지식을 사용해서 사용자 질문에 답변해 주세요.

===
배경지식
{context}

===
사용자 질문
{question}
"""
    )

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def page_ask_my_pdf():
    chain = init_qa_chain()

    query = st.text_input(
        "PDF에 대한 질문을 입력하세요:",
        key="input"
    )

    if query:
        st.markdown("## 답변")
        st.write_stream(chain.stream(query))

def main():
    init_page()

    st.title("PDF QA ♪")

    if "vectorstore" not in st.session_state:
        st.warning("먼저 ▤ Upload PDF(s)에서 PDF 파일을 업로드해 주세요.")
        st.experimental_rerun()
    else:
        page_ask_my_pdf()

if __name__ == "__main__":
    main()
