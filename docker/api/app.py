from flask import Flask, jsonify
import redis
import os
import time

app = Flask(__name__)

# Configurações via variáveis de ambiente (boas práticas de Docker)
REDIS_HOST = os.environ.get('REDIS_HOST', 'redis') 
REDIS_PORT = 6379

# Configuração do Redis com timeout para resistir ao caos
cache = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    socket_connect_timeout=2, # Se o caos travar a rede, desiste em 2s
    socket_timeout=2
)

@app.route('/')
def index():
    try:
        # Simula uma operação de banco de dados
        hits = cache.incr('hits')
        return jsonify({
            "status": "online",
            "message": "Bem-vindo ao sistema resiliente!",
            "visualizacoes": hits
        }), 200
    except redis.exceptions.ConnectionError:
        return jsonify({
            "status": "degradado", 
            "error": "O serviço de cache (Redis) está fora do ar ou sob ataque de caos!"
        }), 503

if __name__ == "__main__":
    # O segredo é o host='0.0.0.0'
    app.run(host='0.0.0.0', port=5000)