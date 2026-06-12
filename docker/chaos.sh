#!/bin/sh

# Lista de containers alvos potenciais
CONTAINERS="api redis worker"

while true; do
  # Sorteia um tempo de espera aleatório entre 30 e 90 segundos antes do próximo ataque
  TEMPO_ESPERA=$(shuf -i 30-90 -n 1)
  echo "[AGENDADOR] Próximo ataque em ${TEMPO_ESPERA} segundos..."
  sleep $TEMPO_ESPERA

  # Sorteia o container alvo
  ALVO=$(echo $CONTAINERS | tr ' ' '\n' | shuf -n 1)
  
  # Sorteia o tipo de anomalia (1 = Kill, 2 = Delay, 3 = Loss)
  TIPO_ANOMALIA=$(shuf -i 1-3 -n 1)

  echo "[CAOS] ⚡ Iniciando ataque contra o container: $ALVO"

  case $TIPO_ANOMALIA in
    1)
      echo "[CAOS] Tipo: Morte Súbita (SIGKILL)"
      docker run --rm -v /var/run/docker.sock:/var/run/docker.sock gaiaadm/pumba:latest \
        --log-level info kill --signal SIGKILL "$ALVO"
      ;;
    2)
      LATENCIA=$(shuf -i 1500-4000 -n 1)
      echo "[CAOS] Tipo: Latência de Rede (${LATENCIA}ms por 20s)"
      docker run --rm --privileged -v /var/run/docker.sock:/var/run/docker.sock gaiaadm/pumba:latest \
        netem --duration 20s --tc-image gaiadocker/iproute2 delay --time "$LATENCIA" "$ALVO"
      ;;
    3)
      PERDA=$(shuf -i 15-40 -n 1)
      echo "[CAOS] Tipo: Perda de Pacotes (${PERDA}% por 20s)"
      docker run --rm --privileged -v /var/run/docker.sock:/var/run/docker.sock gaiaadm/pumba:latest \
        netem --duration 20s --tc-image gaiadocker/iproute2 loss -p "$PERDA" "$ALVO"
      ;;
  esac

  echo "[CAOS] Attack contra $ALVO finalizado. Aguardando próximo ciclo..."
  echo "---------------------------------------------------------"
done