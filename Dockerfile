# Only needed for `docker compose --profile app up`. Day-to-day you run the
# Python on your host against the dockerised Postgres, because rebuilding this
# image to test a one-line change is a bad way to spend your afternoon.
FROM python:3.11-slim

# Unstructured shells out to these. pip cannot install them, and without them
# every scanned PDF fails at OCR with an error that does not mention tesseract.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so the (slow) dependency layer is cached independently
# of application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sift/ ./sift/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY sql/ ./sql/
COPY config/ ./config/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "sift.api:app", "--host", "0.0.0.0", "--port", "8000"]
