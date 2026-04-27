import streamlit as st


def init_page():
    st.set_page_config(
        page_title="Ask My PDF(s)",
        page_icon="☜"
    )


def init_llm_settings():
    st.sidebar.title("LLM 설정")

    models = (
        "GPT-5 mini",
        "GPT-5.1",
        "Claude Sonnet 4.5",
        "Gemini 2.5 Flash"
    )

    selected_model = st.sidebar.selectbox(
        "사용할 LLM 선택",
        models,
        index=0
    )

    st.session_state["selected_model"] = selected_model

    # 선택한 모델에 따라 API Key 입력창만 표시
    if selected_model in ["GPT-5 mini", "GPT-5.1"]:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_api_key", "")
        )
        if api_key:
            st.session_state["openai_api_key"] = api_key

    elif selected_model == "Claude Sonnet 4.5":
        api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.get("anthropic_api_key", "")
        )
        if api_key:
            st.session_state["anthropic_api_key"] = api_key

    elif selected_model == "Gemini 2.5 Flash":
        api_key = st.sidebar.text_input(
            "Google Gemini API Key",
            type="password",
            value=st.session_state.get("google_api_key", "")
        )
        if api_key:
            st.session_state["google_api_key"] = api_key


def main():
    init_page()
    init_llm_settings()

    st.sidebar.success("☜ 왼쪽 메뉴를 진행해 주세요")

    st.markdown(
        """
        - 이 앱에서는 업로드한 PDF에 대해 질문할 수 있습니다.
        - 먼저 왼쪽 메뉴에서 **'▤ Upload PDF(s)'** 를 선택해 PDF를 업로드해 주세요.
        - PDF를 업로드한 뒤에는 **'♬ PDF QA'** 에서 질문해 보세요.
        """
    )


if __name__ == "__main__":
    main()