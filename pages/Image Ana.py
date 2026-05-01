import base64
import os
import streamlit as st

from langchain_openai import ChatOpenAI
from openai import OpenAI


# --------------------------------------------------
# 이미지 프롬프트 템플릿
# --------------------------------------------------
IMAGE_PROMPT_TEMPLATE = """
먼저, 아래 사용자의 요청과 업로드된 이미지를 주의 깊게 읽어주세요.
다음으로, 업로드된 이미지를 기반으로 이미지를 생성해 달라는 사용자의 요청에 따라
이미지 생성용 프롬프트를 작성해 주세요.
프롬프트는 반드시 영어로 작성해야 합니다.

주의:
이미지 속 사람이나 특정 장소, 랜드마크, 상표 등을 식별하지 말아 주세요.
묘사는 사진 속 시각적 요소를 중립적으로 설명하는 방식으로 해주세요.

사용자 입력:
{user_input}

프롬프트에서는 사용자가 업로드한 사진에 무엇이 담겨 있는지,
어떻게 구성되어 있는지를 설명해주세요.
사진의 구도와 줌 정도도 설명해주세요.
사진의 내용을 재현하는 것이 중요합니다.

이미지 생성용 프롬프트를 영어로 출력해주세요.
"""


# --------------------------------------------------
# Sidebar (공통 설정을 session_state에 저장하여 다른 페이지와 중복을 방지)
# --------------------------------------------------
def init_sidebar():
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

    # 모델 선택 (세 페이지에서 공유하는 키 이름인 selected_model 사용)
    selected_model = st.sidebar.selectbox(
        "Model 선택",
        [
            "GPT-5 mini",
            "GPT-5.1",
            "gpt-4o",
            "Claude Sonnet 4.5",
            "Gemini 2.5 Flash",
        ],
        index=["GPT-5 mini", "GPT-5.1", "gpt-4o", "Claude Sonnet 4.5", "Gemini 2.5 Flash"].index(
            st.session_state.get("selected_model", "GPT-5 mini")
        ) if st.session_state.get("selected_model") in ["GPT-5 mini", "GPT-5.1", "gpt-4o", "Claude Sonnet 4.5", "Gemini 2.5 Flash"] else 0
    )

    # Write back to session_state so other pages can use the values
    st.session_state["openai_api_key"] = openai_api_key
    st.session_state["anthropic_api_key"] = anthropic_api_key
    st.session_state["google_api_key"] = google_api_key
    st.session_state["selected_model"] = selected_model

    return openai_api_key, selected_model


# --------------------------------------------------
# OpenAI LLM 생성 (Image prompt 생성을 위한 OpenAI 전용 LLM)
# --------------------------------------------------
def get_openai_llm(openai_api_key, selected_model):
    if not openai_api_key:
        st.info("좌측 사이드바에서 OpenAI API Key를 입력해주세요.")
        st.stop()

    # 이 페이지에서 이미지 프롬프트를 생성하려면 OpenAI 모델을 사용하도록 제한
    if selected_model not in ["GPT-5 mini", "GPT-5.1", "gpt-4o"]:
        st.error("이미지 프롬프트 생성을 위해 OpenAI 모델을 선택해주세요 (예: GPT-5 mini, GPT-5.1, gpt-4o).")
        st.stop()

    model_map = {
        "GPT-5 mini": "gpt-5-mini",
        "GPT-5.1": "gpt-5.1",
        "gpt-4o": "gpt-4o",
    }

    model_name = model_map.get(selected_model, "gpt-4o")

    try:
        os.environ["OPENAI_API_KEY"] = openai_api_key

        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=openai_api_key
        )

        return llm

    except Exception as e:
        st.error(f"LLM 초기화 오류: {str(e)}")
        st.stop()


# --------------------------------------------------
# GPT Image 생성
# --------------------------------------------------
def generate_image(prompt, openai_api_key):
    try:
        client = OpenAI(api_key=openai_api_key)

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="medium",
            n=1
        )

        if not response.data:
            st.error("이미지 생성 결과가 없습니다.")
            return None

        return response.data[0].b64_json

    except Exception as e:
        st.error(f"이미지 생성 오류: {str(e)}")
        return None


# --------------------------------------------------
# 페이지 초기화
# --------------------------------------------------
def init_page():
    st.set_page_config(
        page_title="Image Recognizer",
        page_icon="◈",
        layout="wide"
    )

    st.header("Image Recognizer ◈")


# --------------------------------------------------
# 메인
# --------------------------------------------------
def main():
    init_page()

    # Sidebar (공통 설정 반영)
    openai_api_key, selected_model = init_sidebar()

    # 만약 Upload PDF(s)에서 벡터 DB가 만들어져 있다면 합칠지 물어보기
    vectorstore = st.session_state.get("vectorstore")

    if vectorstore:
        st.info(f"로드된 PDF가 감지되었습니다: {st.session_state.get('document_name', '문서명 없음')}")
        combine_choice = st.radio(
            "로드된 PDF 파일들을 합쳐서 분석할까요?",
            ("합쳐서 분석하기", "이미지 생성으로 진행하기")
        )

        if combine_choice == "합쳐서 분석하기":
            # retriever를 사용해 관련 문서들을 많이 가져와 합치기
            try:
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 150}
                )

                # 빈 쿼리가 허용되지 않을 수 있으므로 일반적인 요약 요청으로 관련 문서 검색
                docs = retriever.get_relevant_documents("전체 문서를 결합하여 분석해줘")

                if not docs:
                    st.warning("벡터 DB에서 문서를 가져오지 못했습니다.")
                else:
                    combined_texts = []
                    for d in docs:
                        # langchain Document 객체의 경우 page_content 속성 사용
                        text = getattr(d, "page_content", None)
                        if not text:
                            # dict 형태일 경우에 대비
                            text = d.get("text") if isinstance(d, dict) else str(d)
                        combined_texts.append(text)

                    combined_text = "\n\n".join(combined_texts)

                    st.markdown("### 결합된 문서(일부)")
                    st.write(combined_text[:5000] + ("..." if len(combined_text) > 5000 else ""))

                    # OpenAI LLM을 이용해 결합된 문서 분석
                    llm = get_openai_llm(openai_api_key, selected_model)

                    st.markdown("### 문서 분석 결과")

                    analysis_prompt = (
                        "다음은 여러 PDF에서 추출한 결합된 텍스트입니다. "
                        "중요한 내용 요약, 주요 키포인트, 가능한 질문 및 주의할 점을 한국어로 정리해주세요.\n\n" + combined_text
                    )

                    # Streaming으로 분석 결과 출력
                    try:
                        analysis_query = [
                            (
                                "user",
                                [
                                    {"type": "text", "text": analysis_prompt}
                                ]
                            )
                        ]

                        st.write_stream(llm.stream(analysis_query))

                    except Exception as e:
                        st.error(f"문서 분석 중 오류가 발생했습니다: {str(e)}")

            except Exception as e:
                st.error(f"벡터 스토어 접근 오류: {str(e)}")

            # 분석을 진행했으므로 이미지 생성 부분은 건너뜁니다.
            return

    # 위에서 PDF 합치기를 선택하지 않았거나 벡터 스토어가 없는 경우 이미지 처리 진행
    uploaded_file = st.file_uploader(
        "이미지를 업로드 해주세요 (Limit. 200MB)",
        type=["png", "jpg", "jpeg", "webp", "gif"]
    )

    if not uploaded_file:
        st.write("먼저 이미지를 업로드해주세요.")
        return

    # 업로드된 원본 이미지 먼저 표시
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    st.markdown("### Original Image")
    st.image(
        file_bytes,
        use_container_width=True
    )

    # 사용자 입력 Form
    with st.form("image_form"):
        user_input = st.text_area(
            "이미지를 어떻게 가공할지 알려주세요!",
            height=150,
            placeholder="예: 배경을 바다로 바꿔주세요"
        )

        submitted = st.form_submit_button(
            "이미지 생성하기"
        )

    if not submitted:
        return

    if not user_input.strip():
        st.warning("이미지 가공 요청을 입력해주세요.")
        return

    # 이미지 base64 처리
    try:
        image_base64 = base64.b64encode(
            file_bytes
        ).decode()

        mime_type = uploaded_file.type
        if not mime_type:
            mime_type = "image/jpeg"

        image_data_url = (
            f"data:{mime_type};base64,{image_base64}"
        )

    except Exception as e:
        st.error(f"파일 처리 오류: {str(e)}")
        return

    # Vision Prompt 생성
    try:
        llm = get_openai_llm(openai_api_key, selected_model)

        query = [
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": IMAGE_PROMPT_TEMPLATE.format(
                            user_input=user_input
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": "auto"
                        }
                    }
                ]
            )
        ]

        st.markdown("### Question")
        st.write(user_input)

        st.markdown("### Image Prompt")

        image_prompt = st.write_stream(
            llm.stream(query)
        )

        if not image_prompt:
            st.error("프롬프트 생성에 실패했습니다.")
            return

    except Exception as e:
        st.error(f"프롬프트 생성 오류: {str(e)}")
        return

    # GPT Image 생성
    with st.spinner("GPT Image가 그림을 그리는 중입니다..."):
        generated_image_base64 = generate_image(
            prompt=image_prompt,
            openai_api_key=openai_api_key
        )

    if not generated_image_base64:
        return

    # 생성된 이미지 마지막에 표시
    try:
        st.markdown("### Generated Image")

        generated_image_bytes = base64.b64decode(
            generated_image_base64
        )

        st.image(
            generated_image_bytes,
            caption=image_prompt,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"결과 출력 오류: {str(e)}")


if __name__ == "__main__":
    main()
