import logging
from logging.handlers import RotatingFileHandler
import time
import random
import sys
import json
import signal
import os
import uuid

# Configuração do diretório e arquivo de logs
LOG_DIR = "k8s-chaos/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app_operation.log")

# Configuração de Logs Estruturados (JSON)
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "trace_id": getattr(record, "trace_id", "N/A"),
            "message": record.getMessage(),
        }
        return json.dumps(log_record)

# Setup do Logger
logger = logging.getLogger("ChaosApp")
logger.setLevel(logging.DEBUG)

# 1. Handler para gravar no arquivo local (rotaciona a cada 10MB, mantém 5 arquivos antigos)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)

# 2. Handler para manter a saída no console (útil para `kubectl logs`)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(JSONFormatter())
logger.addHandler(stdout_handler)

ACTIONS = ["process_payment", "fetch_user_data", "update_cache", "send_email", "connect_db"]

def handle_sigterm(*args):
    logger.info("Sinal SIGTERM recebido. Iniciando graceful shutdown...", extra={"trace_id": str(uuid.uuid4())})
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

def simulate_work():
    logger.info("Aplicação iniciada. Gerando carga de logs...", extra={"trace_id": str(uuid.uuid4())})
    
    while True:
        action = random.choice(ACTIONS)
        status_roll = random.random()
        current_trace = str(uuid.uuid4()) # Trace ID para facilitar parseamento e correlação de eventos

        if status_roll < 0.70:
            logger.info(f"Sucesso ao executar a acao: {action}", extra={"trace_id": current_trace})
        elif status_roll < 0.85:
            logger.debug(f"Detalhes internos de {action} - payload size {random.randint(10, 1000)}kb", extra={"trace_id": current_trace})
        elif status_roll < 0.95:
            logger.warning(f"Lentidao detectada em {action} (duracao {random.uniform(1.0, 5.0):.2f}s)", extra={"trace_id": current_trace})
        else:
            logger.error(f"Falha critica ao executar {action} - Connection Timeout ou Deadlock", extra={"trace_id": current_trace})

        time.sleep(random.uniform(0.1, 0.5))

if __name__ == "__main__":
    try:
        simulate_work()
    except KeyboardInterrupt:
        logger.info("Aplicacao encerrada manualmente (Ctrl+C).", extra={"trace_id": str(uuid.uuid4())})