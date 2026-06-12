FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn aiofiles --no-cache-dir
COPY app.py .
RUN mkdir -p /app/logs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
