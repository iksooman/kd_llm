import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 환경 변수 로드

# Gemini API 키 설정
# .env 파일에 GEMINI_API_KEY=YOUR_API_KEY 형태로 저장해야 합니다.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_response(prompt: str, model_name: str = "gemini-pro"):
    """
    Gemini 모델로부터 응답을 받습니다.
    """
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error getting response from Gemini: {e}")
        return None

if __name__ == "__main__":
    print("Gemini API 연동 테스트를 시작합니다.")
    test_prompt = "안녕하세요, 당신은 누구인가요?"
    response = get_gemini_response(test_prompt)

    if response:
        print("\n--- Gemini 응답 ---")
        print(response)
    else:
        print("\n--- Gemini 응답 실패 ---")
        print("API 키가 올바른지, 인터넷 연결이 되어 있는지 확인해주세요.")
