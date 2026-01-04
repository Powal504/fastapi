from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
device = torch.device("cpu")

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(device)
model.eval()

class ReviewText(BaseModel):
    text: str

@app.post("/predict")
def predict_rating(data: ReviewText):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()  # 0-4
    rating = predicted_class + 1  # rating 1-5

    return {
        "rating": rating,
        "confidence": float(probs[0][predicted_class])
    }
