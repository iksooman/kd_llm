import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

@st.cache_resource
def load_model(model_name_or_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model {model_name_or_path}: {e}")
        return None, None

st.title("LLM Knowledge Distillation Comparison")

st.write("원본 LLM과 증류된 LLM의 답변을 비교해보세요.")

# 모델 로드
original_model_path = "google/gemma-3-1b-it"
distilled_model_path = "./distilled_model"

st.sidebar.header("모델 로딩 상태")

# 증류된 모델 로드 시도
if os.path.exists(distilled_model_path):
    st.sidebar.write(f"증류된 모델 로드 중: {distilled_model_path}")
    distilled_tokenizer, distilled_model = load_model(distilled_model_path)
    if distilled_model:
        st.sidebar.success("증류된 모델 로드 완료!")
    else:
        st.sidebar.error("증류된 모델 로드 실패. 원본 모델로 대체합니다.")
        distilled_tokenizer, distilled_model = load_model(original_model_path) # Fallback
else:
    st.sidebar.warning(f"'{distilled_model_path}' 디렉토리를 찾을 수 없습니다. 원본 모델을 증류된 모델로 간주합니다.")
    distilled_tokenizer, distilled_model = load_model(original_model_path) # Fallback

# 원본 모델 로드
st.sidebar.write(f"원본 모델 로드 중: {original_model_path}")
original_tokenizer, original_model = load_model(original_model_path)
if original_model:
    st.sidebar.success("원본 모델 로드 완료!")
else:
    st.sidebar.error("원본 모델 로드 실패. 애플리케이션을 실행할 수 없습니다.")
    st.stop() # Stop the app if original model cannot be loaded

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("프롬프트를 입력하세요:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if not original_model or not distilled_model:
        st.error("모델 로딩에 문제가 발생했습니다. 다시 시도해주세요.")
    else:
        st.subheader("모델 답변 비교")

        # Prepare messages for models
        messages_for_model = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        # 원본 모델 답변 생성
        with st.spinner("원본 모델 답변 생성 중..."):
            original_chat_template = original_tokenizer.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True)
            original_inputs = original_tokenizer(original_chat_template, return_tensors="pt").to(original_model.device)
            original_outputs = original_model.generate(**original_inputs, max_new_tokens=500, num_return_sequences=1)
            original_response = original_tokenizer.decode(original_outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            original_response = original_response.replace(original_chat_template, "").strip()

            with st.chat_message("original_llm"):
                st.markdown(f"**원본 LLM ({original_model_path}):**")
                st.write(original_response)

        # 증류된 모델 답변 생성
        with st.spinner("""증류된 모델 답변 생성 중...
(증류된 모델이 없으면 원본 모델과 동일한 답변을 생성합니다.)"""):
            distilled_chat_template = distilled_tokenizer.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True)
            distilled_inputs = distilled_tokenizer(distilled_chat_template, return_tensors="pt").to(distilled_model.device)
            distilled_outputs = distilled_model.generate(**distilled_inputs, max_new_tokens=500, num_return_sequences=1)
            distilled_response = distilled_tokenizer.decode(distilled_outputs[0], skip_special_tokens=True)

            # Remove the prompt from the response
            distilled_response = distilled_response.replace(distilled_chat_template, "").strip()

            with st.chat_message("distilled_llm"):
                st.markdown(f"**증류된 LLM ({distilled_model_path if os.path.exists(distilled_model_path) else original_model_path}):**")
                st.write(distilled_response)
        
        # Add assistant responses to chat history
        st.session_state.messages.append({"role": "original_llm", "content": original_response})
        st.session_state.messages.append({"role": "distilled_llm", "content": distilled_response})
