#!/usr/bin/env bash
# fix-chaos-mesh.sh — Diagnostica e reinstala o Chaos Mesh corretamente no Minikube
#
# Uso:
#   ./scripts/fix-chaos-mesh.sh

set -euo pipefail

RED='\033[0;31m'
YLW='\033[1;33m'
GRN='\033[0;32m'
CYN='\033[0;36m'
RST='\033[0m'

log()     { echo -e "${CYN}[$(date '+%H:%M:%S')]${RST} $*"; }
success() { echo -e "${GRN}[$(date '+%H:%M:%S')] ✅ $*${RST}"; }
warn()    { echo -e "${YLW}[$(date '+%H:%M:%S')] ⚠️  $*${RST}"; }
danger()  { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $*${RST}"; }
separator(){ echo -e "${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"; }

separator
log "DIAGNÓSTICO DO AMBIENTE"
separator

# ── 1. Detecta o container runtime do Minikube ────────────────────────────────
log "1/6 — Detectando container runtime..."

RUNTIME=$(minikube ssh "cat /etc/containerd/config.toml 2>/dev/null | grep 'snapshotter' | head -1 || echo 'docker'" 2>/dev/null || echo "unknown")

# Forma mais confiável: verifica qual socket existe
if minikube ssh "test -S /var/run/containerd/containerd.sock" 2>/dev/null; then
  DETECTED_RUNTIME="containerd"
  SOCKET_PATH="/var/run/containerd/containerd.sock"
elif minikube ssh "test -S /var/run/docker.sock" 2>/dev/null; then
  DETECTED_RUNTIME="docker"
  SOCKET_PATH="/var/run/docker.sock"
else
  # Fallback: pergunta ao minikube diretamente
  DETECTED_RUNTIME=$(minikube profile list -o json 2>/dev/null \
    | python3 -c "import sys,json; p=json.load(sys.stdin); print(p['valid'][0]['Config'].get('KubernetesConfig',{}).get('ContainerRuntime','docker'))" 2>/dev/null \
    || echo "docker")
  SOCKET_PATH="/var/run/docker.sock"
  [[ "$DETECTED_RUNTIME" == "containerd" ]] && SOCKET_PATH="/var/run/containerd/containerd.sock"
fi

log "   Runtime detectado: ${YLW}${DETECTED_RUNTIME}${RST}"
log "   Socket: $SOCKET_PATH"

# ── 2. Verifica CRDs existentes ───────────────────────────────────────────────
log "2/6 — Verificando CRDs do Chaos Mesh..."
CRD_COUNT=$(kubectl get crd 2>/dev/null | grep -c "chaos-mesh.org" || echo "0")
log "   CRDs instalados: $CRD_COUNT"

if [[ "$CRD_COUNT" -gt 0 ]]; then
  warn "   CRDs existentes mas possivelmente corrompidos. Removendo para reinstalar limpo..."
  kubectl get crd | grep chaos-mesh.org | awk '{print $1}' | \
    xargs kubectl delete crd --ignore-not-found=true 2>/dev/null || true
  success "   CRDs antigos removidos."
fi

# ── 3. Remove instalação anterior do Helm ────────────────────────────────────
log "3/6 — Verificando instalação Helm anterior..."
if helm status chaos-mesh -n chaos-mesh > /dev/null 2>&1; then
  warn "   Release 'chaos-mesh' encontrado. Removendo..."
  helm uninstall chaos-mesh -n chaos-mesh --wait 2>/dev/null || true
  success "   Release removido."
else
  log "   Nenhuma release anterior encontrada."
fi

# Remove e recria o namespace para garantir estado limpo
kubectl delete namespace chaos-mesh --ignore-not-found=true --wait=true 2>/dev/null || true
kubectl create namespace chaos-mesh
success "   Namespace 'chaos-mesh' recriado."

# ── 4. Instala CRDs manualmente primeiro ─────────────────────────────────────
log "4/6 — Instalando CRDs do Chaos Mesh manualmente..."
kubectl apply -f https://mirrors.chaos-mesh.org/v2.6.3/crd.yaml

log "   Aguardando CRDs ficarem estabelecidos..."
sleep 5

CRD_COUNT=$(kubectl get crd | grep -c "chaos-mesh.org" || echo "0")
if [[ "$CRD_COUNT" -lt 10 ]]; then
  danger "Poucos CRDs instalados ($CRD_COUNT). Algo deu errado."
  exit 1
fi
success "   $CRD_COUNT CRDs instalados com sucesso."

# ── 5. Instala Chaos Mesh via Helm com runtime correto ────────────────────────
log "5/6 — Instalando Chaos Mesh (runtime: $DETECTED_RUNTIME)..."

helm repo add chaos-mesh https://charts.chaos-mesh.org 2>/dev/null || true
helm repo update

helm install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace=chaos-mesh \
  --version 2.6.3 \
  --set chaosDaemon.runtime="$DETECTED_RUNTIME" \
  --set chaosDaemon.socketPath="$SOCKET_PATH" \
  --set dashboard.securityMode=false \
  --wait \
  --timeout 120s

success "Chaos Mesh instalado."

# ── 6. Verifica saúde dos pods ────────────────────────────────────────────────
log "6/6 — Verificando pods do Chaos Mesh..."
sleep 5
kubectl get pods -n chaos-mesh

NOT_READY=$(kubectl get pods -n chaos-mesh --no-headers | grep -v "Running\|Completed" | wc -l | tr -d ' ')
if [[ "$NOT_READY" -gt 0 ]]; then
  warn "$NOT_READY pod(s) ainda não estão Running. Aguardando mais 30s..."
  sleep 30
  kubectl get pods -n chaos-mesh
fi

separator
success "CHAOS MESH PRONTO!"
echo ""
echo "  Teste rápido — aplique um experimento:"
echo "    kubectl apply -f chaos/01-pod-kill.yaml"
echo ""
echo "  Dashboard visual:"
echo "    kubectl port-forward -n chaos-mesh svc/chaos-dashboard 2333:2333"
echo "    → http://localhost:2333"
echo ""
echo "  Agora rode o chaos-runner normalmente:"
echo "    ./scripts/chaos-runner.sh 120"
separator
