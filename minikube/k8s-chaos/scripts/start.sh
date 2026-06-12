#!/usr/bin/env bash
# start.sh — Inicia o ambiente completo e abre todas as interfaces
#
# O que faz:
#   1. Garante que o Minikube está rodando
#   2. Sobe port-forwards do app e do Chaos Mesh em background
#   3. Detecta WSL2 / Linux / macOS e abre o navegador no endereço certo
#   4. Inicia o chaos-runner em background (com logs em arquivo)
#   5. Mostra status ao vivo dos pods
#
# Uso:
#   ./scripts/start.sh                  # chaos-runner com intervalo padrão (120s)
#   ./scripts/start.sh --interval 60    # intervalo customizado em segundos
#   ./scripts/start.sh --no-chaos       # sobe só as interfaces, sem chaos-runner
#   ./scripts/start.sh --no-browser     # não abre navegador automaticamente

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_DIR/logs"

# ── Flags ─────────────────────────────────────────────────────────────────────
INTERVAL=120
NO_CHAOS=false
NO_BROWSER=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --no-chaos)   NO_CHAOS=true;   shift ;;
    --no-browser) NO_BROWSER=true; shift ;;
    *) shift ;;
  esac
done

# ── Cores ─────────────────────────────────────────────────────────────────────
CYN='\033[0;36m'; GRN='\033[0;32m'; YLW='\033[1;33m'; RST='\033[0m'
log()     { echo -e "${CYN}[$(date '+%H:%M:%S')]${RST} $*"; }
success() { echo -e "${GRN}[$(date '+%H:%M:%S')] ✅ $*${RST}"; }
warn()    { echo -e "${YLW}[$(date '+%H:%M:%S')] ⚠️  $*${RST}"; }
separator(){ echo -e "${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"; }

# ── PID file para cleanup ──────────────────────────────────────────────────────
PID_FILE="$PROJECT_DIR/.start.pids"
> "$PID_FILE"   # limpa pids anteriores

cleanup_on_exit() {
  echo ""
  log "Encerrando processos em background..."
  while IFS= read -r PID; do
    kill "$PID" 2>/dev/null || true
  done < "$PID_FILE"
  rm -f "$PID_FILE"
  log "Ambiente parado. Para retomar: ./scripts/start.sh"
}
trap cleanup_on_exit SIGINT SIGTERM

# ── Detecta como abrir o navegador ────────────────────────────────────────────
open_browser() {
  local URL="$1"
  if $NO_BROWSER; then return; fi

  # WSL2
  if grep -qi microsoft /proc/version 2>/dev/null; then
    # No WSL2, usa o host_ip para chegar ao port-forward do WSL
    local HOST_IP
    HOST_IP=$(hostname -I | awk '{print $1}')
    local WSL_URL="${URL/localhost/$HOST_IP}"
    cmd.exe /c start "$WSL_URL" 2>/dev/null || true
    log "   Navegador Windows aberto em: $WSL_URL"
    return
  fi

  # macOS
  if [[ "$(uname)" == "Darwin" ]]; then
    open "$URL" 2>/dev/null || true
    return
  fi

  # Linux nativo
  xdg-open "$URL" 2>/dev/null || \
  sensible-browser "$URL" 2>/dev/null || \
  warn "Não foi possível abrir o navegador automaticamente. Acesse: $URL"
}

# ── Inicia um port-forward em background ──────────────────────────────────────
start_port_forward() {
  local NAME="$1"
  local NAMESPACE="$2"
  local SVC="$3"
  local PORT_MAP="$4"   # ex: "8080:80"
  local LOCAL_PORT="${PORT_MAP%%:*}"
  local LOGFILE="$LOGS_DIR/portforward-${NAME}.log"

  mkdir -p "$LOGS_DIR"

  # Mata qualquer port-forward anterior na mesma porta
  fuser -k "${LOCAL_PORT}/tcp" 2>/dev/null || true
  sleep 1

  log "   Iniciando port-forward: $NAME ($PORT_MAP)..."
  kubectl port-forward -n "$NAMESPACE" "svc/$SVC" "$PORT_MAP" \
    >> "$LOGFILE" 2>&1 &
  local PID=$!
  echo "$PID" >> "$PID_FILE"

  # Aguarda a porta responder (até 15s)
  local TRIES=0
  while ! nc -z localhost "$LOCAL_PORT" 2>/dev/null; do
    sleep 1
    TRIES=$((TRIES + 1))
    if [[ $TRIES -ge 15 ]]; then
      warn "Port-forward $NAME demorou mais que o esperado. Verifique: $LOGFILE"
      return
    fi
  done
  success "Port-forward $NAME pronto em localhost:$LOCAL_PORT"
}

# ─────────────────────────────────────────────────────────────────────────────
separator
log "🚀 k8s-chaos — START"
separator

# ── 1. Minikube ────────────────────────────────────────────────────────────────
log "1/4 — Verificando Minikube..."
if ! minikube status | grep -q "Running"; then
  log "   Minikube não está rodando. Iniciando..."
  minikube start --cpus=2 --memory=4096
fi
success "Minikube rodando."

# Garante que o kubectl aponta pro contexto certo
kubectl config use-context minikube > /dev/null 2>&1 || true

# ── 2. Verifica se o app está deployado ───────────────────────────────────────
log "2/4 — Verificando deploy do app..."
if ! kubectl get deployment demo-app -n default > /dev/null 2>&1; then
  warn "Deployment 'demo-app' não encontrado. Execute ./scripts/setup.sh primeiro."
  exit 1
fi

# Aguarda pods ficarem prontos
kubectl rollout status deployment/demo-app --timeout=60s > /dev/null
success "App demo-app: $(kubectl get pods -l app=demo-app --no-headers | grep -c Running || echo 0) pod(s) Running."

# ── 3. Port-forwards ──────────────────────────────────────────────────────────
log "3/4 — Iniciando port-forwards..."
start_port_forward "app"        "default"     "demo-app-svc"  "8080:80"
start_port_forward "chaos-mesh" "chaos-mesh"  "chaos-dashboard" "2333:2333"

# ── 4. Abre os navegadores ────────────────────────────────────────────────────
log "4/4 — Abrindo interfaces no navegador..."
sleep 1
open_browser "http://localhost:8080"
open_browser "http://localhost:2333"

# ── 5. Chaos runner (opcional) ────────────────────────────────────────────────
if ! $NO_CHAOS; then
  mkdir -p "$LOGS_DIR"
  RUNNER_LOG="$LOGS_DIR/chaos-runner.log"
  log "▶  Iniciando chaos-runner (intervalo: ${INTERVAL}s)..."
  log "   Log do runner: $RUNNER_LOG"
  bash "$SCRIPT_DIR/chaos-runner.sh" "$INTERVAL" >> "$RUNNER_LOG" 2>&1 &
  RUNNER_PID=$!
  echo "$RUNNER_PID" >> "$PID_FILE"
  success "chaos-runner iniciado (PID $RUNNER_PID)."
fi

# ── Resumo ────────────────────────────────────────────────────────────────────
separator
echo ""
echo -e "  ${GRN}App demo:${RST}       http://localhost:8080"
echo -e "  ${GRN}Chaos Mesh:${RST}     http://localhost:2333"
if ! $NO_CHAOS; then
echo -e "  ${GRN}Chaos runner:${RST}   intervalo de ${INTERVAL}s  |  log: logs/chaos-runner.log"
fi
echo ""
echo -e "  Endpoints do app:"
echo -e "    /          → info do pod"
echo -e "    /health    → health check"
echo -e "    /stress    → stress de CPU"
echo ""
echo -e "  ${YLW}Pressione Ctrl+C para parar tudo.${RST}"
separator
echo ""

# ── Mantém o script vivo mostrando status dos pods ────────────────────────────
log "Monitorando pods (atualiza a cada 15s)..."
echo ""
while true; do
  echo -e "${CYN}── Pods ── $(date '+%H:%M:%S') ────────────────────────────${RST}"
  kubectl get pods -l app=demo-app \
    --no-headers \
    -o custom-columns='POD:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[0].ready,RESTARTS:.status.containerStatuses[0].restartCount' \
    2>/dev/null || true
  sleep 15
done
