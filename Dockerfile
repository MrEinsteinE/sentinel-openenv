FROM ghcr.io/meta-pytorch/openenv-base:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SENTINEL_HOST=0.0.0.0
ENV SENTINEL_PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=180 -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:7860/health', timeout=3).read(); sys.exit(0)" || exit 1

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
