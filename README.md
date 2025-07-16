# Gemini와 Llama 3.2 1B를 이용한 Knowledge Distillation

이 프로젝트는 Google의 강력한 상용 LLM인 **Gemini**를 교사 모델(Teacher Model)로, 경량 오픈소스 모델인 **Llama 3.2 1B**를 학생 모델(Student Model)로 사용하여 Knowledge Distillation (지식 증류)을 수행하는 것을 목표로 합니다.

<br>

✨ **Gemini CLI를 통해 개발되었습니다.**

이 프로젝트의 모든 코드는 Google의 **Gemini CLI**를 통해 작성되고 관리되었습니다. 파일 생성, 코드 수정, Git 커밋, Dockerfile 작성 등 개발의 전 과정이 Gemini와의 대화를 통해 진행되었습니다. 이는 LLM을 활용한 새로운 차원의 개발 워크플로우를 보여주는 예시입니다.

<br>

## 프로젝트 목표

*   **교사 모델 (Gemini):** 고품질의 질문-답변 쌍(Synthetic Data)을 생성합니다.
*   **학생 모델 (Llama 3.2 1B):** 교사 모델이 생성한 데이터를 학습하여, 적은 비용으로도 높은 성능을 내는 경량 모델을 만듭니다.
*   **재현성:** Docker를 통해 누구나 동일한 환경에서 프로젝트를 실행하고 학습을 재현할 수 있도록 합니다.

<br>

## 프로젝트 구조

*   `data_generator.py`: Gemini API를 호출하여 지식 증류에 사용할 합성 데이터를 생성합니다. (`synthetic_qa_data.jsonl` 파일로 저장)
*   `distillation_trainer.py`: 생성된 데이터를 사용하여 Llama 3.2 1B 모델을 fine-tuning(미세 조정)하는 학습 스크립트입니다.
*   `requirements.txt`: 프로젝트에 필요한 Python 라이브러리 목록입니다.
*   `Dockerfile`: 재현 가능한 학습 환경을 구축하기 위한 Docker 설정 파일입니다.

<br>

## 실행 방법

### 1. 프로젝트 클론 및 환경 설정

```bash
git clone <repository_url>
cd kd_project
pip install -r requirements.txt
```

### 2. Gemini API 키 설정

프로젝트 루트 디렉터리에 `.env` 파일을 생성하고, 그 안에 자신의 Gemini API 키를 입력합니다.

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

### 3. 합성 데이터 생성

다음 명령어를 실행하여 Gemini로부터 질문-답변 데이터를 생성합니다.

```bash
python data_generator.py
```

### 4. 모델 학습 (GPU 환경 필요)

데이터 생성이 완료되면, 다음 명령어로 Knowledge Distillation 학습을 시작합니다.

```bash
python distillation_trainer.py
```
학습이 완료되면 `distilled_model` 디렉터리에 결과물이 저장됩니다.

<br>

## Docker를 이용한 실행

로컬 환경의 복잡한 설정 없이, Docker를 사용하여 간편하게 프로젝트를 실행할 수 있습니다.

### 1. Docker 이미지 빌드

```bash
docker build -t kd-llm-app .
```

### 2. Docker 컨테이너 실행

`--gpus all` 옵션을 통해 컨테이너가 GPU를 사용하도록 설정하고, `-v .:/app`으로 현재 디렉터리를 마운트하며, `--env-file .env`로 API 키를 안전하게 전달합니다.

```bash
docker run --gpus all -it -v .:/app --env-file .env kd-llm-app
```

컨테이너 내부에서 `data_generator.py`와 `distillation_trainer.py`를 순서대로 실행하세요.

<br>

## 다음 단계

*   클라우드 플랫폼(Kaggle 또는 Hugging Face Spaces)에서 실제 증류 학습 진행
*   학습 결과 분석 및 모델 성능 평가
*   하이퍼파라미터 튜닝 및 데이터셋 확장
