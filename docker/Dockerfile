FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    DATA_DIR=/app/data_pdf \
    STORE_DIR=/app/vector_store \
    OLLAMA_URL=http://host.docker.internal:11434

WORKDIR /app

# Install dependencies first (better layer caching)
COPY source_code/requirements.txt .
RUN pip install -r requirements.txt

# Application code (build context = project root)
COPY source_code/ .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
