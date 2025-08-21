FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY models ./models

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8000"]
