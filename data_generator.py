import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_response(prompt: str, model_name: str = "gemini-2.5-pro", max_retries: int = 3, delay: int = 5):
    """
    Gemini 모델로부터 응답을 받습니다. API 호출 실패 시 재시도 로직 포함.
    """
    print("모델 생성")
    model = genai.GenerativeModel(model_name)
    for attempt in range(max_retries):
        try:
            print("데이터 전송")
            response = model.generate_content(prompt)
            print(response)
            return response.text
        except Exception as e:
            print(f"Error getting response from Gemini (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay) # 재시도 전 지연
    return None

def generate_qa_pairs(num_pairs: int = 10, topic: str = "general knowledge", output_file: str = "synthetic_qa_data.jsonl"):
    """
    Gemini를 사용하여 질문-답변 쌍을 생성하고 JSONL 파일로 저장합니다.
    """
    print(f"{num_pairs}개의 질문-답변 쌍을 생성합니다. 주제: {topic}")
    qa_data = []
    for i in range(num_pairs):
        prompt = f"Generate a unique and interesting question about {topic} and its concise answer. Format it as a JSON object with 'question' and 'answer' keys. Example: {{\"question\": \"What is the capital of France?\", \"answer\": \"Paris.\"}}"

        response_text = get_gemini_response(prompt)
        if response_text:
            try:
                # Gemini가 JSON 형식으로 응답하도록 프롬프트했지만, 때로는 추가 텍스트가 붙을 수 있습니다.
                # JSON 파싱을 위해 첫 번째와 마지막 중괄호 사이의 문자열만 추출합니다.
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx : end_idx + 1]
                    qa_pair = json.loads(json_str)
                    if "question" in qa_pair and "answer" in qa_pair:
                        qa_data.append(qa_pair)
                        print(f"Generated {len(qa_data)}/{num_pairs} pairs.")
                    else:
                        print(f"Skipping invalid JSON structure: {json_str}")
                else:
                    print(f"Skipping response not containing valid JSON: {response_text}")
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e} - 원본 응답: {response_text}")
        else:
            print(f"질문-답변 쌍 생성 실패 (시도 {i+1})")
        time.sleep(1) # API 호출 간 지연을 두어 Rate Limit 방지

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in qa_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"\n{len(qa_data)}개의 질문-답변 쌍이 '{output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    # 테스트를 위해 적은 수의 데이터를 생성합니다.
    # 실제 학습에는 더 많은 데이터가 필요합니다.
    generate_qa_pairs(num_pairs=5, topic="science and technology", output_file="synthetic_qa_data.jsonl")
