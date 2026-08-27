#!/usr/bin/env bash
# chaos-runner.sh
# Coleta logs de todos os pods continuamente e roda os experimentos de chaos
# em rotação. Cada experimento é aplicado, aguardado e removido antes do próximo.
#
# Uso:
#   ./scripts/chaos-runner.sh [intervalo_entre_experimentos_em_segundos]
#
# Exemplo (roda um experimento a cada 3 minutos):
#   ./scripts/chaos-runner.sh 180

set -euo pipefail

# ─── Configurações ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_DIR/logs_appficticio"
CHAOS_DIR="$PROJECT_DIR/chaos"
NAMESPACE="default"
INTERVAL="${1:-120}"   # segundos entre experimentos (padrão: 2 minutos)

# Ordem dos experimentos (arquivo → tipo do recurso)
declare -A CHAOS_RESOURCES=(
  ["01-pod-kill.yaml"]="podchaos/pod-kill-demo"
  ["02-network-delay.yaml"]="networkchaos/network-delay-demo"
  ["03-cpu-stress.yaml"]="stresschaos/cpu-stress-demo"
  ["04-pod-kill-chaoslog.yaml"]="podchaos/pod-kill-chaoslog"
  ["05-network-delay-chaoslog.yaml"]="networkchaos/network-delay-chaoslog"
  ["06-cpu-stress-chaoslog.yaml"]="stresschaos/cpu-stress-chaoslog"
)
CHAOS_ORDER=("01-pod-kill.yaml" "02-network-delay.yaml" "03-cpu-stress.yaml" \
             "04-pod-kill-chaoslog.yaml" "05-network-delay-chaoslog.yaml" "06-cpu-stress-chaoslog.yaml")

# ─── Funções utilitárias ───────────────────────────────────────────────────────
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

separator() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ─── Coleta de logs de todos os pods em background ────────────────────────────
start_log_collectors() {
  mkdir -p "$LOGS_DIR"
  log "📁 Pasta de logs: $LOGS_DIR"

  # Inicia um coletor em background para cada pod atual
  for POD in $(kubectl get pods -n "$NAMESPACE" -l 'app in (demo-app,chaos-log-app)' -o jsonpath='{.items[*].metadata.name}'); do
    local LOGFILE="$LOGS_DIR/${POD}.log"
    if ! pgrep -f "kubectl logs.*$POD" > /dev/null 2>&1; then
      log "📝 Coletando logs do pod: $POD → $LOGFILE"
      kubectl logs -n "$NAMESPACE" "$POD" -f --timestamps >> "$LOGFILE" 2>&1 &
    fi
  done
}

# Reinicia coletores para pods novos (ex: após pod-kill)
refresh_log_collectors() {
  start_log_collectors
}

# ─── Limpeza de experimentos órfãos ──────────────────────────────────────────
cleanup_all_experiments() {
  log "🧹 Limpando experimentos ativos..."
  kubectl delete podchaos    --all -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
  kubectl delete networkchaos --all -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
  kubectl delete stresschaos  --all -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null || true
}

# ─── Aplica e aguarda um experimento ─────────────────────────────────────────
run_experiment() {
  local FILE="$1"
  local RESOURCE="${CHAOS_RESOURCES[$FILE]}"
  local FULL_PATH="$CHAOS_DIR/$FILE"
  local EXPERIMENT_LOG="$LOGS_DIR/experiments.log"

  separator
  log "🔥 INICIANDO EXPERIMENTO: $FILE"
  log "   Recurso: $RESOURCE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXPERIMENT_START file=$FILE resource=$RESOURCE" >> "$EXPERIMENT_LOG"

  # Aplica o experimento
  kubectl apply -f "$FULL_PATH" -n "$NAMESPACE"

  # Descobre a duração do experimento no YAML
  local DURATION_STR
  DURATION_STR=$(grep 'duration:' "$FULL_PATH" | awk '{print $2}' | tr -d '"' | head -1)
  local DURATION_SEC=60  # fallback

  if [[ "$DURATION_STR" =~ ^([0-9]+)s$ ]]; then
    DURATION_SEC="${BASH_REMATCH[1]}"
  elif [[ "$DURATION_STR" =~ ^([0-9]+)m$ ]]; then
    DURATION_SEC=$(( ${BASH_REMATCH[1]} * 60 ))
  fi

  log "⏱️  Duração do experimento: ${DURATION_STR} (${DURATION_SEC}s)"
  log "👀 Monitorando pods durante o experimento..."

  # Monitora os pods a cada 10s enquanto o experimento roda
  local ELAPSED=0
  while [ "$ELAPSED" -lt "$DURATION_SEC" ]; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    local SNAPSHOT_FILE="$LOGS_DIR/pods-snapshot-$(date '+%Y%m%d_%H%M%S').log"
    kubectl get pods -n "$NAMESPACE" -l 'app in (demo-app,chaos-log-app)' \
      --no-headers \
      -o custom-columns='NAME:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[0].ready,RESTARTS:.status.containerStatuses[0].restartCount' \
      >> "$SNAPSHOT_FILE" 2>&1 || true
    log "   [${ELAPSED}s/${DURATION_SEC}s] snapshot → $(basename "$SNAPSHOT_FILE")"

    # Reinicia coletores para pods que possam ter sido recriados
    refresh_log_collectors
  done

  # Remove o experimento
  log "🗑️  Removendo experimento: $RESOURCE"
  kubectl delete -f "$FULL_PATH" -n "$NAMESPACE" --ignore-not-found=true
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXPERIMENT_END file=$FILE resource=$RESOURCE" >> "$EXPERIMENT_LOG"
  log "✅ Experimento concluído: $FILE"
}

# ─── Salva estado final dos pods ──────────────────────────────────────────────
save_final_pod_state() {
  local STATE_FILE="$LOGS_DIR/pod-state-$(date '+%Y%m%d_%H%M%S').log"
  kubectl get pods -n "$NAMESPACE" -l 'app in (demo-app,chaos-log-app)' -o wide >> "$STATE_FILE" 2>&1 || true
  log "💾 Estado dos pods salvo em: $(basename "$STATE_FILE")"
}

# ─── Signal handler (Ctrl+C limpa tudo) ───────────────────────────────────────
cleanup_on_exit() {
  echo ""
  log "🛑 Interrompido. Limpando experimentos ativos..."
  cleanup_all_experiments
  # Para todos os kubectl logs em background
  pkill -f "kubectl logs" 2>/dev/null || true
  save_final_pod_state
  log "👋 Saindo."
  exit 0
}
trap cleanup_on_exit SIGINT SIGTERM

# ─── Main loop ────────────────────────────────────────────────────────────────
main() {
  separator
  log "🚀 chaos-runner iniciado"
  log "   Intervalo entre experimentos: ~${INTERVAL}s (aleatório, 50%-150%)"
  log "   Namespace: $NAMESPACE"
  log "   Pasta de logs: $LOGS_DIR"
  separator

  # Garante que não há experimentos ativos de execuções anteriores
  cleanup_all_experiments

  # Inicia coletores de log
  start_log_collectors

  local CONTADOR=0
  while true; do
    CONTADOR=$((CONTADOR + 1))

    # Intervalo aleatório (50%-150% do INTERVAL configurado) em vez de fixo —
    # evita que o motor de detecção aprenda a periodicidade do caos como padrão.
    local TEMPO_ESPERA
    TEMPO_ESPERA=$(shuf -i $((INTERVAL / 2))-$((INTERVAL * 3 / 2)) -n 1)
    log "⏳ Aguardando ${TEMPO_ESPERA}s antes do próximo experimento (aleatório)..."
    sleep "$TEMPO_ESPERA"

    # Garante que os coletores estão rodando (pods podem ter mudado)
    refresh_log_collectors

    # Experimento sorteado — não mais em ordem fixa 01→02→03
    local FILE
    FILE=$(printf '%s\n' "${CHAOS_ORDER[@]}" | shuf -n 1)

    log "🎲 === EXPERIMENTO #$CONTADOR (sorteado: $FILE) ==="
    run_experiment "$FILE"
  done
}

main
