from flask import Flask, render_template_string, jsonify
import redis
import os
import time
from datetime import datetime
import logging
import sys

# 1. Configuração de Log para o Fluentd (STDOUT)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chaos-logger")

app = Flask(__name__)

REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
cache = redis.Redis(host=REDIS_HOST, port=6379, socket_connect_timeout=1, socket_timeout=1)

# 2. FUNÇÃO ÚNICA (Envia para Tela + Fluentd)
def add_event_log(msg, level="INFO"):
    # Envia para o Fluentd via console
    if level == "ERROR":
        logger.error(f"CAOS: {msg}")
    elif level == "WARNING":
        logger.warning(f"CAOS: {msg}")
    else:
        logger.info(f"CAOS: {msg}")
    
    try:
        # Salva no Redis para mostrar na tela
        timestamp = datetime.now().strftime("%H:%M:%S")
        cache.lpush("chaos_logs", f"[{timestamp}] {msg}")
        cache.ltrim("chaos_logs", 0, 4)
    except:
        pass

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Chaos Dashboard TCC</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body { font-family: 'Segoe UI', sans-serif; text-align: center; transition: background 0.5s; padding: 20px; background-color: {{ color }}; color: white; }
        .card { background: rgba(0,0,0,0.5); padding: 20px; border-radius: 15px; display: inline-block; min-width: 400px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .status { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
        .metrics { display: flex; justify-content: space-around; margin: 20px 0; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px; }
        .log-container { background: #222; color: #0f0; text-align: left; padding: 15px; border-radius: 5px; height: 150px; overflow-y: auto; font-family: monospace; font-size: 0.9em; margin-top: 20px; border: 1px solid #444; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Monitor de Resiliência</h1>
        <div class="status">{{ status }}</div>
        <p>{{ message }}</p>
        <div class="metrics">
            <div>Latência:<br><strong>{{ latency }}ms</strong></div>
            <div>Visualizações:<br><strong>{{ hits }}</strong></div>
            <div>Horário:<br><span>{{ current_time }}</span></div>
        </div>
        <div class="log-container">
            <strong>Histórico de Eventos:</strong><br>
            {% for log in event_logs %}
                <div>> {{ log }}</div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    start_time = time.time()
    event_logs = []
    
    try:
        hits = cache.incr('hits')
        raw_logs = cache.lrange("chaos_logs", 0, -1)
        event_logs = [l.decode('utf-8') for l in raw_logs]
        status = "SISTEMA SAUDÁVEL"
        message = "Aguardando injeção de caos..."
        color = "#2ecc71" # Verde
    except Exception as e:
        hits = "ERR"
        status = "FALHA DE CONEXÃO"
        message = "O banco de dados (Redis) está inacessível!"
        color = "#e74c3c" # Vermelho
        event_logs = ["CRITICAL: Conexão com Redis perdida!"]
        add_event_log("FALHA CRÍTICA: Conexão com Redis perdida", level="ERROR")

    latency = round((time.time() - start_time) * 1000, 2)
    
    if latency > 500 and hits != "ERR":
        color = "#f39c12" # Laranja
        status = "LATÊNCIA DETECTADA"
        message = f"Alerta: Rede instável ({latency}ms)"
        add_event_log(f"LATÊNCIA ALTA: {latency}ms detectados", level="WARNING")

    return render_template_string(HTML_TEMPLATE, status=status, hits=hits, 
                                 latency=latency, color=color, message=message,
                                 current_time=datetime.now().strftime("%H:%M:%S"),
                                 event_logs=event_logs)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
