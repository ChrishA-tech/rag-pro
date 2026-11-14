FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY rag_pro ./rag_pro
COPY data ./data
RUN pip install --no-cache-dir .
RUN rag-pro ingest data/sample_docs --index /app/.index
ENV RAG_INDEX_DIR=/app/.index
EXPOSE 8000
CMD ["uvicorn", "rag_pro.api:app", "--host", "0.0.0.0", "--port", "8000"]
