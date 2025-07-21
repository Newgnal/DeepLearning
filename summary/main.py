from fastapi import FastAPI
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch

app = FastAPI(
    title="KoBART 요약 API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

MODEL_PATH = "./kobart_summary_ft"
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

class SummaryRequest(BaseModel):
    text: str

def generate_summary(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.post("/summarize")
def summarize(request: SummaryRequest):
    result = generate_summary(request.text)
    return {"summary": result}

# @app.get("/") 리디렉션 코드 삭제!


