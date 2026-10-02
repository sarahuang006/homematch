FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
ENV PYTHONPATH=/app/src

EXPOSE 8000
# Build the vector index on first start (needs OPENAI_API_KEY at runtime, never baked into the image)
CMD ["sh", "-c", "python scripts/build_index.py --if-missing && uvicorn homematch.api:app --host 0.0.0.0 --port 8000"]
