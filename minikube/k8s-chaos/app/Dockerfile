FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn aiofiles --no-cache-dir
COPY app.py .

# /app/logs é o ponto de montagem esperado para o volume externo
# (ex.: hostPath do k8s mapeado para ./k8s-chaos/logs no host).
# LOG_DIR pode ser sobrescrito em runtime se precisar apontar para outro caminho.
ENV LOG_DIR=/app/logs
RUN mkdir -p /app/logs

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
