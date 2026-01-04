FROM python:3.11-alpine

# Alpine ma minimalny rozmiar, ale musimy zainstalować build-base i libc6-compat dla torch
RUN apk add --no-cache \
    build-base \
    curl \
    ca-certificates \
    libc6-compat \
    && rm -rf /var/cache/apk/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "ml_api:app", "--host", "0.0.0.0", "--port", "8000"]
