FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY ml_api.py .

CMD ["sh", "-c", "uvicorn ml_api:app --host 0.0.0.0 --port ${PORT:-8000}"]

