FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAINMOSAIC_DATA_DIR=/data \
    RAINMOSAIC_PORT=8000

WORKDIR /srv

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py ./

RUN useradd --system --uid 10001 --create-home --home-dir /home/appuser appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /srv

USER appuser
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["python", "main.py"]
