import redis
import time

# Use o nome do serviço 'redis' definido no docker-compose
r = redis.Redis(host='redis', port=6379, decode_responses=True)

while True:
    try:
        # blpop aguarda até ter algo na lista 'tarefas'
        # Isso evita que o worker fique fritando o CPU e o Redis
        tarefa = r.blpop("tarefas", timeout=5)
        if tarefa:
            print(f"Processando: {tarefa[1]}")
        else:
            print("Aguardando tarefas...")
    except Exception as e:
        print(f"Erro na conexão: {e}")
        time.sleep(2) # Espera antes de tentar de novo