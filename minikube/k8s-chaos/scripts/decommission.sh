#!/usr/bin/env bash
# decommission.sh — Remove TUDO relacionado ao ambiente k8s-chaos
#
# O que este script faz (em ordem):
#   1. Para o chaos-runner se estiver rodando
#   2. Remove todos os experimentos de chaos ativos
#   3. Remove o app (deployment + service)
#   4. Desinstala o Chaos Mesh via Helm
#   5. Remove os namespaces criados
#   6. (Opcional) Destrói o cluster Minikube completamente
#   7. (Opcional) Arquiva ou apaga a pasta logs/
#
# Uso:
#   ./scripts/decommission.sh              # modo interativo (pergunta antes de cada etapa destrutiva)
#   ./scripts/decommission.sh --full       # remove tudo incluindo Minikube, sem perguntar
#   ./scripts/decommission.sh --keep-logs  # preserva a pasta logs/ mesmo no modo --full

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_DIR/logs"
NAMESPACE="default"

FULL_MODE=false
KEEP_LOGS=false

for ARG in "$@"; do
  case "$ARG" in
    --full)       FULL_MODE=true ;;
    --keep-logs)  KEEP_LOGS=true ;;
  esac
done

# ─── Utilitários ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
YLW='\033[1;33m'
GRN='\033[0;32m'
CYN='\033[0;36m'
RST='\033[0m'

log()     { echo -e "${CYN}[$(date '+%H:%M:%S')]${RST} $*"; }
success() { echo -e "${GRN}[$(date '+%H:%M:%S')] ✅ $*${RST}"; }
warn()    { echo -e "${YLW}[$(date '+%H:%M:%S')] ⚠️  $*${RST}"; }
danger()  { echo -e "${RED}[$(date '+%H:%M:%S')] 🔴 $*${RST}"; }
separator(){ echo -e "${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"; }

# Pergunta sim/não (retorna 0=sim, 1=não)
ask() {
  local PROMPT="$1"
  if $FULL_MODE; then return 0; fi
  echo -en "${YLW}$PROMPT [s/N]: ${RST}"
  read -r RESP
  [[ "$RESP" =~ ^[sS]$ ]]
}

# Verifica se o kubectl consegue falar com o cluster
cluster_alive() {
  kubectl cluster-info > /dev/null 2>&1
}

# ─── Etapas ───────────────────────────────────────────────────────────────────

step_stop_chaos_runner() {
  separator
  log "ETAPA 1 — Parando o chaos-runner..."

  local PIDS
  PIDS=$(pgrep -f "chaos-runner.sh" 2>/dev/null || true)
  if [[ -n "$PIDS" ]]; then
    echo "$PIDS" | xargs kill -SIGTERM 2>/dev/null || true
    sleep 2
    success "chaos-runner.sh encerrado (PID(s): $PIDS)"
  else
    log "chaos-runner.sh não estava rodando."
  fi

  # Para também qualquer kubectl logs em background
  pkill -f "kubectl logs" 2>/dev/null || true
  log "Coletores de log encerrados."
}

step_remove_chaos_experiments() {
  separator
  log "ETAPA 2 — Removendo experimentos de Chaos Mesh ativos..."

  if ! cluster_alive; then
    warn "Cluster não acessível — pulando remoção de experimentos."
    return
  fi

  local FOUND=false

  for KIND in podchaos networkchaos stresschaos iochaos httpchaos dnschaos; do
    local COUNT
    COUNT=$(kubectl get "$KIND" -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$COUNT" -gt 0 ]]; then
      FOUND=true
      log "  Removendo $COUNT $KIND..."
      kubectl delete "$KIND" --all -n "$NAMESPACE" --ignore-not-found=true
    fi
  done

  if $FOUND; then
    success "Todos os experimentos removidos."
  else
    log "Nenhum experimento ativo encontrado."
  fi
}

step_remove_app() {
  separator
  log "ETAPA 3 — Removendo o app demo-app..."

  if ! cluster_alive; then
    warn "Cluster não acessível — pulando remoção do app."
    return
  fi

  kubectl delete deployment demo-app   -n "$NAMESPACE" --ignore-not-found=true
  kubectl delete service   demo-app-svc -n "$NAMESPACE" --ignore-not-found=true

  # Aguarda os pods terminarem
  log "  Aguardando pods encerrarem..."
  kubectl wait --for=delete pod -l app=demo-app -n "$NAMESPACE" --timeout=60s 2>/dev/null || true

  success "App removido."
}

step_uninstall_chaos_mesh() {
  separator
  log "ETAPA 4 — Desinstalando Chaos Mesh..."

  if ! cluster_alive; then
    warn "Cluster não acessível — pulando desinstalação do Chaos Mesh."
    return
  fi

  if helm status chaos-mesh -n chaos-mesh > /dev/null 2>&1; then
    helm uninstall chaos-mesh -n chaos-mesh
    success "Helm release 'chaos-mesh' removido."
  else
    log "Helm release 'chaos-mesh' não encontrado — já foi removido ou nunca instalado."
  fi

  # Remove os CRDs do Chaos Mesh
  log "  Removendo CRDs do Chaos Mesh..."
  kubectl get crd | grep chaos-mesh.org | awk '{print $1}' | \
    xargs kubectl delete crd --ignore-not-found=true 2>/dev/null || true

  # Remove o namespace
  if kubectl get namespace chaos-mesh > /dev/null 2>&1; then
    kubectl delete namespace chaos-mesh --ignore-not-found=true
    log "  Namespace 'chaos-mesh' removido."
  fi

  success "Chaos Mesh completamente desinstalado."
}

step_delete_minikube() {
  separator
  danger "ETAPA 5 — Destruir o cluster Minikube"
  warn "  ⚠️  Esta ação é IRREVERSÍVEL. O cluster e todos os dados serão apagados."

  if ask "  Deseja destruir o cluster Minikube?"; then
    minikube delete
    success "Cluster Minikube destruído."
  else
    log "  Mantendo o cluster Minikube. Você pode pará-lo com: minikube stop"
  fi
}

step_handle_logs() {
  separator
  log "ETAPA 6 — Pasta de logs: $LOGS_DIR"

  if [[ ! -d "$LOGS_DIR" ]] || [[ -z "$(ls -A "$LOGS_DIR" 2>/dev/null)" ]]; then
    log "  Pasta de logs vazia ou inexistente — nada a fazer."
    return
  fi

  local LOG_COUNT
  LOG_COUNT=$(find "$LOGS_DIR" -type f | wc -l | tr -d ' ')
  local LOG_SIZE
  LOG_SIZE=$(du -sh "$LOGS_DIR" 2>/dev/null | cut -f1)
  log "  Encontrados $LOG_COUNT arquivo(s) — $LOG_SIZE"

  if $KEEP_LOGS; then
    log "  --keep-logs ativo: pasta preservada em $LOGS_DIR"
    return
  fi

  if $FULL_MODE; then
    # No modo full, arquiva automaticamente
    local ARCHIVE="$PROJECT_DIR/logs-$(date '+%Y%m%d_%H%M%S').tar.gz"
    tar -czf "$ARCHIVE" -C "$PROJECT_DIR" logs/
    rm -rf "$LOGS_DIR"
    success "Logs arquivados em: $ARCHIVE"
    return
  fi

  echo ""
  echo "  O que deseja fazer com os logs?"
  echo "    [1] Arquivar em .tar.gz e remover a pasta"
  echo "    [2] Manter a pasta logs/ intacta"
  echo "    [3] Apagar sem arquivar"
  echo -en "${YLW}  Escolha [1/2/3]: ${RST}"
  read -r CHOICE

  case "$CHOICE" in
    1)
      local ARCHIVE="$PROJECT_DIR/logs-$(date '+%Y%m%d_%H%M%S').tar.gz"
      tar -czf "$ARCHIVE" -C "$PROJECT_DIR" logs/
      rm -rf "$LOGS_DIR"
      success "Logs arquivados em: $ARCHIVE"
      ;;
    3)
      rm -rf "$LOGS_DIR"
      success "Pasta logs/ removida."
      ;;
    *)
      log "Pasta logs/ mantida em: $LOGS_DIR"
      ;;
  esac
}

# ─── Resumo final ─────────────────────────────────────────────────────────────
print_summary() {
  separator
  success "Decommission concluído!"
  echo ""
  echo "  O que foi removido:"
  echo "    • chaos-runner.sh e coletores de log encerrados"
  echo "    • Experimentos de Chaos Mesh (PodChaos, NetworkChaos, StressChaos)"
  echo "    • Deployment e Service do demo-app"
  echo "    • Helm release chaos-mesh + CRDs + namespace chaos-mesh"
  if $FULL_MODE; then
    echo "    • Cluster Minikube destruído"
  fi
  echo ""
  if cluster_alive 2>/dev/null; then
    echo "  O cluster Minikube ainda está rodando."
    echo "  Para pará-lo:    minikube stop"
    echo "  Para destruí-lo: minikube delete"
  fi
  separator
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  separator
  danger "k8s-chaos — DECOMMISSION"
  if $FULL_MODE; then
    warn "Modo: --full (sem confirmações, tudo será removido)"
  else
    log "Modo: interativo"
  fi
  separator
  echo ""

  step_stop_chaos_runner
  step_remove_chaos_experiments
  step_remove_app
  step_uninstall_chaos_mesh

  if $FULL_MODE; then
    step_delete_minikube
  else
    if ask "Deseja também destruir o cluster Minikube? (etapa final)"; then
      step_delete_minikube
    else
      log "Pulando destruição do Minikube."
    fi
  fi

  step_handle_logs
  print_summary
}

main
