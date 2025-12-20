FROM python:3.11 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip wheel -r requirements.txt -w /wheels

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install /wheels/*
EXPOSE 8501
COPY . .
