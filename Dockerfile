# NVIDIA CUDA 런타임을 부모 이미지로 사용합니다.
# 이 이미지는 GPU 가속 작업에 필요한 CUDA 툴킷과 cuDNN을 포함하고 있습니다.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 설치 중 대화형 프롬프트를 방지하기 위해 환경 변수를 설정합니다.
ENV DEBIAN_FRONTEND=noninteractive

# Python, pip, git을 설치합니다.
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# 컨테이너의 작업 디렉토리를 설정합니다.
WORKDIR /app

# 의존성 파일을 작업 디렉토리로 복사합니다.
COPY requirements.txt .

# requirements.txt에 명시된 패키지들을 설치합니다.
# --no-cache-dir 옵션은 이미지 크기를 줄여줍니다.
RUN pip3 install --no-cache-dir -r requirements.txt

# 나머지 애플리케이션 코드를 작업 디렉토리로 복사합니다.
COPY . .

# 컨테이너가 시작될 때 실행할 기본 명령어를 설정합니다.
# 수동으로 스크립트를 실행할 수 있는 대화형 셸을 제공합니다.
CMD ["bash"]

