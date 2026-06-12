#!/usr/bin/env bash
# setup.sh — Prepara o ambiente completo do zero
# Uso: ./setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "1/5 — Verificando Minikube..."
if ! minikube status | grep -q "Running"; then
  log "    Iniciando Minikube..."
  minikube start --cpus=2 --memory=4096
else
  log "    Minikube já está rodando."
fi

log "2/5 — Apontando Docker para o daemon do Minikube..."
eval "$(minikube docker-env)"

log "3/5 — Build da imagem demo-app:latest..."
docker build -t demo-app:latest "$PROJECT_DIR/app/"

log "4/5 — Aplicando deployment e service..."
kubectl apply -f "$PROJECT_DIR/k8s/deployment.yaml"
kubectl rollout status deployment/demo-app --timeout=90s

log "5/5 — Instalando Chaos Mesh (se necessário)..."
if ! helm status chaos-mesh -n chaos-mesh > /dev/null 2>&1; then
  kubectl create namespace chaos-mesh --dry-run=client -o yaml | kubectl apply -f -
  helm repo add chaos-mesh https://charts.chaos-mesh.org 2>/dev/null || true
  helm repo update
  helm install chaos-mesh chaos-mesh/chaos-mesh \
    --namespace=chaos-mesh \
    --set chaosDaemon.runtime=docker \
    --set chaosDaemon.socketPath=/var/run/docker.sock \
    --version 2.6.3
  log "    Aguardando Chaos Mesh ficar pronto..."
  kubectl rollout status deployment/chaos-controller-manager -n chaos-mesh --timeout=120s
else
  log "    Chaos Mesh já instalado."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅  Setup concluído!"
echo ""
echo "  Pods rodando:"
kubectl get pods -l app=demo-app
echo ""
echo "  Para expor o app localmente:"
echo "    kubectl port-forward svc/demo-app-svc 8080:80"
echo ""
echo "  Para iniciar o chaos runner (experimento a cada 2min):"
echo "    ./scripts/chaos-runner.sh 120"
echo ""
echo "  Para abrir o dashboard do Chaos Mesh:"
echo "    kubectl port-forward -n chaos-mesh svc/chaos-dashboard 2333:2333"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
