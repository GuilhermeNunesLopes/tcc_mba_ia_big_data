
cd k8s-chaos

chmod +x scripts/decommission.sh
# Modo interativo — pergunta antes de cada etapa destrutiva
./scripts/decommission.sh

# Modo full — remove absolutamente tudo sem perguntar
./scripts/decommission.sh --full

# Remove tudo mas preserva a pasta logs/
./scripts/decommission.sh --full --keep-logs