from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. 라벨 정의
theme_labels = [
    "반도체·AI", "2차전지·친환경 에너지", "헬스케어·바이오", "IT·인터넷 서비스",
    "금융·보험", "소비재·엔터테인먼트", "자동차·모빌리티", "방산·항공우주",
    "부동산·리츠", "채권·금리", "환율·외환", "원자재·귀금속", "기타"
]
id2label = {i: label for i, label in enumerate(theme_labels)}

MODEL_PATH = "./saved_model_etf"

# (A) 모델 로드 (초기 1회)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

app = FastAPI(title="ETF 뉴스 테마 분류 API", version="0.1.0")

class NewsRequest(BaseModel):
    text: str

def predict_theme(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

# ✅ 루트: 200 OK
@app.get("/", tags=["meta"])
def root():
    return {
        "service": "ETF Theme Classification",
        "version": "0.1.0",
        "predict_endpoint": "/predict",
        "health": "/health",
        "docs": "/docs"
    }

# ✅ 헬스체크: 모니터링/로드밸런서용
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}

@app.post("/predict", tags=["inference"])
def get_theme(req: NewsRequest):
    label = predict_theme(req.text)
    return {"theme": label}
