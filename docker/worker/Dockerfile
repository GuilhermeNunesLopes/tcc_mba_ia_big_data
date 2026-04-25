FROM python:3.11-slim

WORKDIR /app

# Instala a biblioteca do Redis
RUN pip install --no-cache-dir redis

# Copia o código do worker
COPY worker.py .

# Comando para rodar o worker
CMD ["python", "worker.py"]