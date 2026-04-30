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
# Sidebar
# --------------------------------------------------
def init_sidebar():
    st.sidebar.title("⚙️ Options")

    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    model_name = st.sidebar.selectbox(
        "Model 선택",
        [
            "gpt-4o",
            "gpt-4.1",
            "gpt-5.1"
        ]
    )

    return openai_api_key, model_name


# --------------------------------------------------
# OpenAI LLM 생성
# --------------------------------------------------
def get_llm(openai_api_key, model_name):
    if not openai_api_key:
        st.info("좌측 사이드바에서 OpenAI API Key를 입력해주세요.")
        st.stop()

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

    # Sidebar
    openai_api_key, model_name = init_sidebar()

    # LLM 생성
    llm = get_llm(
        openai_api_key=openai_api_key,
        model_name=model_name
    )

    uploaded_file = st.file_uploader(
        "이미지를 업로드 해주세요 (Limit. 200MB)",
        type=["png", "jpg", "jpeg", "webp", "gif"]
    )

    if not uploaded_file:
        st.write("먼저 이미지를 업로드해주세요.")
        return

    # --------------------------------------------------
    # 업로드된 원본 이미지 먼저 표시
    # --------------------------------------------------
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    st.markdown("### Original Image")
    st.image(
        file_bytes,
        use_container_width=True
    )

    # --------------------------------------------------
    # 사용자 입력 Form
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 이미지 base64 처리
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Vision Prompt 생성
    # --------------------------------------------------
    try:
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

    # --------------------------------------------------
    # GPT Image 생성
    # --------------------------------------------------
    with st.spinner("GPT Image가 그림을 그리는 중입니다..."):
        generated_image_base64 = generate_image(
            prompt=image_prompt,
            openai_api_key=openai_api_key
        )

    if not generated_image_base64:
        return

    # --------------------------------------------------
    # 생성된 이미지 마지막에 표시
    # --------------------------------------------------
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