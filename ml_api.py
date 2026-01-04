from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Powal/roberta-review-sentiment"

device = torch.device("cpu")

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            subfolder="model_roberta"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            subfolder="model_roberta"
        )
        model.to(device)
        model.eval()

app = FastAPI(title="RoBERTa Review Sentiment API")

class ReviewText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: ReviewText):
    load_model()

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
