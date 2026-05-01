import base64
import os
import streamlit as st

from openai import OpenAI
from langchain_openai import ChatOpenAI

# 공통 사이드바 및 모델 선택을 shared.py로 분리하여 중복 제거
from shared import init_sidebar, select_model

# 이미지 프롬프트 템플릿
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


def get_openai_llm_for_image(openai_api_key: str, selected_model: str):
    """이미지 프롬프트 생성을 위해 OpenAI LLM을 반환합니다. 여기서는 OpenAI 모델만 허용합니다."""
    if not openai_api_key:
        st.info("좌측 사이드바에서 OpenAI API Key를 입력해주세요.")
        st.stop()

    # 이미지 프롬프트 생성을 위해 OpenAI 모델을 사용하도록 제한
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


def init_page():
    st.set_page_config(
        page_title="Image Recognizer",
        page_icon="◈",
        layout="wide"
    )

    st.header("Image Recognizer ◈")


def main():
    init_page()

    # Sidebar 초기화 (shared.py의 값을 사용)
    openai_api_key, selected_model = init_sidebar()

    # 만약 Upload PDF(s)에서 벡터 DB가 만들어져 있다면 합칠지 물어보기
    vectorstore = st.session_state.get("vectorstore")

    if vectorstore:
        document_name = st.session_state.get('document_name', '문서명 없음')
        st.info(f"로드된 PDF가 감지되었습니다: {document_name}")

        combine_choice = st.radio(
            "로드된 PDF 파일들을 합쳐서 분석할까요?",
            ("합쳐서 분석하기", "이미지 생성으로 진행하기")
        )

        if combine_choice == "합쳐서 분석하기":
            try:
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 150}
                )

                docs = retriever.get_relevant_documents("전체 문서를 결합하여 분석해줘")

                if not docs:
                    st.warning("벡터 DB에서 문서를 가져오지 못했습니다.")
                else:
                    combined_texts = []
                    for d in docs:
                        text = getattr(d, "page_content", None)
                        if not text:
                            text = d.get("text") if isinstance(d, dict) else str(d)
                        combined_texts.append(text)

                    combined_text = "\n\n".join(combined_texts)

                    st.markdown("### 결합된 문서 (미리보기)")
                    st.write(combined_text[:5000] + ("..." if len(combined_text) > 5000 else ""))

                    # shared.select_model을 사용하여 선택된 LLM으로 분석
                    try:
                        llm = select_model()

                        st.markdown("### 문서 분석 결과 (요약 및 주요 포인트)")

                        analysis_prompt = (
                            "다음은 여러 PDF에서 추출한 결합된 텍스트입니다. "
                            "중요한 내용 요약, 주요 키포인트, 가능한 질문 및 주의할 점을 한국어로 정리해주세요.\n\n" + combined_text
                        )

                        res = llm.generate([analysis_prompt])

                        analysis_text = ""
                        for gen_list in res.generations:
                            for gen in gen_list:
                                text = getattr(gen, "text", None) or str(gen)
                                analysis_text += text

                        st.text_area("문서 분석", value=analysis_text, height=400)

                    except Exception as e:
                        st.error(f"문서 분석 중 오류가 발생했습니다: {str(e)}")

            except Exception as e:
                st.error(f"벡터 스토어 접근 오류: {str(e)}")

            # 분석을 진행했으므로 이미지 생성 부분은 건너뜁니다.
            return

    # 이미지 업로드 및 처리
    uploaded_file = st.file_uploader(
        "이미지를 업로드 해주세요 (Limit. 200MB)",
        type=["png", "jpg", "jpeg", "webp", "gif"]
    )

    if not uploaded_file:
        st.write("먼저 이미지를 업로드해주세요.")
        return

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()

    st.markdown("### Original Image")
    st.image(
        file_bytes,
        use_container_width=True
    )

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
        llm = get_openai_llm_for_image(openai_api_key, selected_model)

        prompt_text = IMAGE_PROMPT_TEMPLATE.format(user_input=user_input) + "\nImage URL: " + image_data_url

        st.markdown("### Question")
        st.write(user_input)

        st.markdown("### Image Prompt (영문)")

        res = llm.generate([prompt_text])

        image_prompt = ""
        for gen_list in res.generations:
            for gen in gen_list:
                image_prompt += getattr(gen, "text", str(gen))

        image_prompt = image_prompt.strip()

        if not image_prompt:
            st.error("프롬프트 생성에 실패했습니다.")
            return

        st.code(image_prompt)

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
