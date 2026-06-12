# k8s-chaos — Demo App + Chaos Engineering

App FastAPI com 3 pods rodando no Minikube, com rotação automática de
experimentos de Chaos Mesh e coleta contínua de logs.

## Estrutura

```
k8s-chaos/
├── app/
│   ├── app.py          # FastAPI app
│   └── Dockerfile
├── k8s/
│   └── deployment.yaml # Deployment (3 replicas) + Service
├── chaos/
│   ├── 01-pod-kill.yaml        # Mata um pod aleatório (30s)
│   ├── 02-network-delay.yaml   # Latência de rede 200ms (60s)
│   └── 03-cpu-stress.yaml      # Stress de CPU 80% (45s)
├── scripts/
│   ├── setup.sh        # Setup completo do zero
│   └── chaos-runner.sh # Rotação automática + coleta de logs
└── logs/               # Gerado automaticamente
    ├── demo-app-xxxxx.log        # Log de cada pod
    ├── experiments.log           # Histórico de experimentos
    └── pods-snapshot-*.log       # Estado dos pods durante chaos
```

## Início rápido

```bash
# 1. Setup completo (Minikube + build + deploy + Chaos Mesh)
chmod +x scripts/setup.sh scripts/chaos-runner.sh
./scripts/setup.sh

# 2. Inicia o chaos runner (experimento a cada 2 minutos)
./scripts/chaos-runner.sh 120

# Em outro terminal — acessa o app
kubectl port-forward svc/demo-app-svc 8080:80
curl http://localhost:8080/health

# Dashboard do Chaos Mesh
kubectl port-forward -n chaos-mesh svc/chaos-dashboard 2333:2333
# → http://localhost:2333
```

## chaos-runner.sh — como funciona

O script entra em loop infinito e, a cada rodada:

1. Aguarda N segundos (configurável, padrão 120s)
2. Aplica o experimento `01-pod-kill.yaml` e monitora os pods a cada 10s
3. Quando o experimento termina, remove o recurso do cluster
4. Repete para `02-network-delay.yaml` e `03-cpu-stress.yaml`
5. Reinicia o ciclo

Paralelamente, um coletor `kubectl logs -f` roda em background para cada
pod. Quando um pod é morto e recriado, o runner detecta o novo pod e inicia
um novo coletor automaticamente.

### Ajustar o intervalo

```bash
# 5 minutos entre experimentos
./scripts/chaos-runner.sh 300

# 30 segundos (para testes rápidos)
./scripts/chaos-runner.sh 30
```

## Logs gerados

| Arquivo | Conteúdo |
|---|---|
| `logs/demo-app-<hash>.log` | Logs contínuos de cada pod |
| `logs/experiments.log` | Timestamp de início/fim de cada experimento |
| `logs/pods-snapshot-<ts>.log` | Estado dos pods a cada 10s durante chaos |

## Parar tudo

`Ctrl+C` no chaos-runner já remove os experimentos ativos e salva o estado
final dos pods. Para destruir o cluster:

```bash
minikube delete
```
