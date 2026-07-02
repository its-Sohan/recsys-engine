FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY frontend/ frontend/
COPY artifacts/ artifacts/
COPY pyproject.toml .

RUN pip install -e . --no-deps --quiet

EXPOSE 8000

CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8000", "--server.address", "0.0.0.0", "--server.headless", "true"]
