# CUDA 11.8과 Python 3.10이 포함된 NVIDIA 공식 이미지 사용
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /workspace

# 패키지 업데이트 및 필수 패키지 설치 (tmux 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux \
    git \
    python3.10 \
    python3-pip \
    python3.10-dev \ 
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip 최신 버전으로 업그레이드
RUN pip install --upgrade pip

# PyTorch, torchvision, torchaudio 설치 (CUDA 11.8 지원)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 기본 셸로 bash 실행
CMD ["bash"]
