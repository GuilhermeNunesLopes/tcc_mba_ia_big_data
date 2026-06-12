#PRE REQUISITOS: Virtualização Habilitada, Docker Desktop, WSL2,
mini=$(minikube version --short 2>/dev/null || echo "minikube not found")

if [[ "$mini" == "minikube not found" ]]; then
  curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
  sudo install minikube-linux-amd64 /usr/local/bin/minikube
fi

if kubectl version --client > /dev/null 2>&1; then
  echo "kubectl já instalado."
else
 curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
 sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
fi        

# Inicia o cluster com 2 CPUs e 4GB de RAM
minikube start --cpus=2 --memory=4096

# Verifica se está rodando
minikube status
kubectl cluster-info

minikube dashboard &  
  

#app
# 1. Descompacte e dê permissão em todos os scripts de uma vez
unzip k8s-chaos.zip && cd k8s-chaos
chmod +x scripts/*.sh

# 2. Setup completo — só na primeira vez
./scripts/setup.sh

# 3. Se der erro de CRD (como aconteceu), roda o fix — só se necessário
./scripts/fix-chaos-mesh.sh

# 4. Inicia tudo — interfaces + chaos runner
./scripts/start.sh