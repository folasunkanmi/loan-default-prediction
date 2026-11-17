# Multi-stage-friendly single image for API + optional Streamlit
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && pip install fastapi uvicorn[standard] pydantic>=2.0 httpx

COPY . .

# Train on synthetic data at build time so the container always has a model
RUN python -m src.data.generate_data && python -m src.models.train

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
