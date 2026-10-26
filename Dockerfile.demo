# Lightweight Dockerfile for StockAI Terminal Demo Edition
# Optimized for DevOps evaluation, Minikube Kubernetes, and Render
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    APP_MODE=demo

# Minimal OS dependencies: curl for health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with UID 10001 for security best practices
RUN groupadd -r appgroup --gid 10001 && \
    useradd -r -g appgroup --uid 10001 --create-home --home-dir /home/appuser appuser

WORKDIR /app

# Install minimal demo dependencies (no Ollama, FAISS, PyTorch, Pandas)
COPY requirements-demo.txt ./
RUN pip install --no-cache-dir --default-timeout=100 -r requirements-demo.txt

# Copy only demo application and static assets
COPY demo_server.py ./
COPY demo_data ./demo_data
COPY static ./static

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Fast, non-blocking health check probe
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/live || exit 1

CMD ["python", "demo_server.py"]
