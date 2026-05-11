FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy package source and install
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Copy inference artifacts (model + tokenizer only — see .dockerignore)
COPY artifacts/model_trainer/ artifacts/model_trainer/

# Copy app and templates
COPY app.py .
COPY templates/ templates/
COPY config/ config/
COPY params.yaml .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
