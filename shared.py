import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# 공통 사이드바 초기화 및 세션 상태 관리
def init_sidebar():
    # 이미 초기화 되어 있으면 기존값 반환
    if "sidebar_initialized" in st.session_state:
        return st.session_state.get("openai_api_key", ""), st.session_state.get("selected_model", "GPT-5 mini")

    st.sidebar.title("⚙️ Options")

    # API Keys
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        value=st.session_state.get("openai_api_key", "")
    )

    anthropic_api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="an-...",
        value=st.session_state.get("anthropic_api_key", "")
    )

    google_api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        placeholder="AIza...",
        value=st.session_state.get("google_api_key", "")
    )

    # 모델 선택
    model_options = [
        "GPT-5 mini",
        "GPT-5.1",
        "gpt-4o",
        "Claude Sonnet 4.5",
        "Gemini 2.5 Flash",
    ]

    default_index = 0
    if st.session_state.get("selected_model") in model_options:
        default_index = model_options.index(st.session_state.get("selected_model"))

    selected_model = st.sidebar.selectbox(
        "Model 선택",
        model_options,
        index=default_index
    )

    # Write back to session_state so other pages can use the values
    st.session_state["openai_api_key"] = openai_api_key
    st.session_state["anthropic_api_key"] = anthropic_api_key
    st.session_state["google_api_key"] = google_api_key
    st.session_state["selected_model"] = selected_model
    st.session_state["sidebar_initialized"] = True

    return openai_api_key, selected_model


# 공통 모델 선택 함수 (PDF 분석이나 QA에서 사용)
def select_model(temperature=0):
    selected_model = st.session_state.get("selected_model")

    if selected_model == "GPT-5 mini":
        api_key = st.session_state.get("openai_api_key")

        if not api_key:
            st.error("OpenAI API Key를 입력해주세요.")
            st.stop()

        return ChatOpenAI(
            temperature=temperature,
            model="gpt-5-mini",
            api_key=api_key
        )

    elif selected_model == "GPT-5.1":
        api_key = st.session_state.get("openai_api_key")

        if not api_key:
            st.error("OpenAI API Key를 입력해주세요.")
            st.stop()

        return ChatOpenAI(
            temperature=temperature,
            model="gpt-5.1",
            api_key=api_key
        )

    elif selected_model == "Claude Sonnet 4.5":
        api_key = st.session_state.get("anthropic_api_key")

        if not api_key:
            st.error("Anthropic API Key를 입력해주세요.")
            st.stop()

        return ChatAnthropic(
            temperature=temperature,
            model="claude-sonnet-4-5-20250929",
            api_key=api_key
        )

    elif selected_model == "Gemini 2.5 Flash":
        api_key = st.session_state.get("google_api_key")

        if not api_key:
            st.error("Google Gemini API Key를 입력해주세요.")
            st.stop()

        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model="gemini-2.5-flash",
            google_api_key=api_key
        )

    else:
        st.error("LLM을 먼저 선택해주세요.")
        st.stop()
