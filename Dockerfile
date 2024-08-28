# Ubuntu 24.04 베이스 이미지 사용
FROM ubuntu:24.04

# 기본 패키지, Python 3.10, MySQL 클라이언트 라이브러리 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Miniforge3 설치
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh

# PATH에 conda 추가
ENV PATH="/opt/conda/bin:$PATH"

# 환경 파일 복사 및 의존성 설치
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean -afy && \
    conda run -n NAGNAE-AI pip install mysqlclient

# 프로젝트 파일 복사
COPY . /app

# conda 환경 활성화
RUN echo "conda activate NAGNAE-AI" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# 포트 설정
EXPOSE 8000

# FastAPI 애플리케이션 실행
CMD ["conda", "run", "-n", "NAGNAE-AI", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
