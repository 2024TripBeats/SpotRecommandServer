# 베이스 이미지로 Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일을 컨테이너로 복사
COPY ./requirements.txt /app/requirements.txt
COPY ./data /app/data
COPY . /app

# 의존성 설치
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# FastAPI 애플리케이션이 실행될 포트 설정
EXPOSE 8001

# 애플리케이션 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
