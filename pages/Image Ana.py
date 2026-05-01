import base64
import os
import streamlit as st

from openai import OpenAI

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


def _extract_text_from_llm_result(res):
    """범용적으로 LLM 결과에서 텍스트를 추출합니다.
    가능한 형태:
    - 문자열
    - res.generations (list[list[Generation]])
      - Generation.text
      - Generation.message.content
      - Generation can be a str
    - dict / list 등
    """
    # 문자열인 경우 그대로 반환
    if isinstance(res, str):
        return res

    # LLMResult 형태
    if hasattr(res, "generations"):
        texts = []
        try:
            for gen_list in res.generations:
                for gen in gen_list:
                    # gen이 문자열일 수 있음
                    if isinstance(gen, str):
                        texts.append(gen)
                        continue

                    text = None

                    # 우선 text 속성 확인 (가장 일반적)
                    if hasattr(gen, "text") and gen.text:
                        # gen.text가 리스트나 기타 타입일 수 있으므로 안전하게 변환
                        if isinstance(gen.text, (list, tuple)):
                            text = "\n".join([str(t) for t in gen.text])
                        else:
                            text = str(gen.text)

                    # message 형태인지 확인
                    elif hasattr(gen, "message") and gen.message:
                        message = gen.message
                        # message가 dict인지, 객체인지, 문자열인지 모두 안전하게 처리
                        if isinstance(message, dict):
                            # dict에 content 또는 text가 있을 수 있음
                            text = message.get("content") or message.get("text") or str(message)
                        else:
                            # 객체일 경우 attribute로 접근하되 안전하게
                            text = getattr(message, "content", None) or getattr(message, "text", None) or str(message)

                    # 드물게 content 속성을 바로 가지고 있을 수 있음
                    elif hasattr(gen, "content"):
                        text = getattr(gen, "content")

                    else:
                        text = str(gen)

                    if text is not None:
                        texts.append(text)
        except Exception:
            # 구조가 예상과 다르면 문자열 변환으로 처리
            return str(res)

        return "\n".join(texts).strip()

    # 리스트나 dict 등 기타 타입 처리
    if isinstance(res, list):
        return "\n".join([str(r) for r in res]).strip()

    if isinstance(res, dict):
        for key in ("content", "text", "message"):
            if key in res:
                val = res[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    return val.get("content") or val.get("text") or str(val)
                return str(val)
        return str(res)

    return str(res)


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
                        combined_text.append(text) if False else combined_texts.append(text)

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

                        analysis_text = _extract_text_from_llm_result(res)

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
        # PDF QA.py와 동일하게 shared.select_model을 사용하여 LLM과 API 키를 처리합니다.
        # select_model은 st.session_state에서 선택된 모델과 API 키를 확인하고 적절한 LLM 인스턴스를 반환합니다.
        llm = select_model()

        prompt_text = IMAGE_PROMPT_TEMPLATE.format(user_input=user_input) + "\nImage URL: " + image_data_url

        st.markdown("### PROMPT")
        st.write(prompt_text)

        st.markdown("### Image Prompt (영문)")

        res = llm.generate([prompt_text])
        
        image_prompt = _extract_text_from_llm_result(res)

        image_prompt = image_prompt.strip()
        st.markdown("### IMAGE PROMPT")
        st.write(image_prompt)

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
