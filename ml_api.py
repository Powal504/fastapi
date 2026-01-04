from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
device = torch.device("cpu")

app = FastAPI(title="DistilBERT Sentiment API")

tokenizer = None
model = None

class ReviewText(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

@app.post("/predict")
def predict_sentiment(data: ReviewText):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()

    return {
        "rating": predicted_class + 1,
        "confidence": float(torch.max(probs).item())
    }
