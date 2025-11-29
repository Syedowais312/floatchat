# -------- Stage 1: Builder --------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip config set global.timeout 600 && \
    pip wheel --no-cache-dir -r requirements.txt -w /wheels

# -------- Stage 2: Runtime --------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . .

EXPOSE 5000
CMD ["python", "query.py"]
    