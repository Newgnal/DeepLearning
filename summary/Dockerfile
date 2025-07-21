# 1. Python 3.10-slim 이미지를 기반으로 설정
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 복사 후 패키지 설치
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4. 현재 디렉토리의 모든 코드 복사
COPY . .

# 5. FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]