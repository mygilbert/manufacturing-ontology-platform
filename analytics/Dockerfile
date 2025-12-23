# ============================================================
# Manufacturing Ontology Platform - Analytics Engine
# ============================================================
FROM python:3.12-slim

# 작업 디렉토리
WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ /app/src/

# 모델 디렉토리
RUN mkdir -p /app/models

# 환경 변수
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV MODEL_SAVE_PATH=/app/models

# 볼륨 (모델 저장용)
VOLUME /app/models

# 실행 (기본: 분석 서비스)
CMD ["python", "-u", "src/main.py"]
