"""
Fonte única das pastas de log conhecidas pelo motor de detecção.
Evita duplicar a mesma lista em main_v6.py e modules/dashboard.py.
"""

#PASTAS_DISPONIVEIS = [
#    "logs_filtrados",
#    "docker/logs_appficticio",
#    "minikube/k8s-chaos/logs_appficticio",
#    "experimento/test1",
#    "experimento/test2",
#]

PASTAS_DISPONIVEIS = [
  "docker/logs_appficticio",
  "minikube/k8s-chaos/logs_appficticio"
]

#PASTAS_DISPONIVEIS =[
#    "logs_filtrados"
#]