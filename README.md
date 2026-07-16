Aqui está o arquivo pronto para você copiar e colar no seu `README.md`:

```markdown
# Detecção de Anomalias em Logs para Root Cause em Microsserviços

![AIOps](https://img.shields.io/badge/Focus-AIOps-blueviolet)
![Machine Learning](https://img.shields.io/badge/ML-Unsupervised-green)
![SRE](https://img.shields.io/badge/Area-SRE%20%26%20DevOps-orange)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

## 📋 Sobre o Projeto

Este projeto apresenta uma abordagem de **Aprendizado Não Supervisionado** aplicada a logs de aplicações em arquiteturas de microsserviços. O objetivo é automatizar a identificação de eventos anômalos para acelerar a análise de causa raiz (*Root Cause Analysis*), transformando a gestão de incidentes de reativa para proativa, eliminando a dependência de regras estáticas.

## 🚀 Motivação e Contexto

Ambientes modernos de microsserviços oferecem resiliência e escalabilidade, mas geram um volume de dados (logs) humanamente impossível de analisar manualmente em tempo real. 

### O Problema
* **Complexidade:** Milhares de sub-ambientes gerando logs simultâneos.
* **Métricas em Risco:** O aumento no tempo de análise manual eleva o **MTTI** (Tempo Médio de Identificação) e o **MTTR** (Tempo Médio de Resolução).
* **Fadiga de Alertas:** Equipes de SRE e DevOps sobrecarregadas por notificações excessivas ou regras estáticas que não acompanham a evolução do sistema.

### A Solução
Utilizar técnicas de **AIOps** para detectar anomalias sem a necessidade de rótulos prévios. Como sistemas dinâmicos apresentam padrões de falha em constante evolução, o pipeline de aprendizado não supervisionado permite isolar comportamentos incomuns e correlacioná-los de forma visual, agilizando o diagnóstico da falha em cascata.

## 🧠 Arquitetura do Pipeline Analítico

O motor de inferência foi projetado com foco em alta performance e redução de ruído, operando no seguinte fluxo:

1. **Ingestão e Parsing (Drain3):** Extração automatizada de templates de logs, transformando texto bruto semi-estruturado em eventos categorizados, mascarando variáveis dinâmicas (IPs, Hexadecimais).
2. **Vetorização (TF-IDF):** Transformação dos templates de logs em matrizes numéricas esparsas, capturando a frequência e a relevância térmica dos termos na infraestrutura.
3. **Redução de Dimensionalidade (TruncatedSVD):** Compressão da matriz esparsa em um espaço denso, otimizando o processamento computacional e destacando os padrões de variância primários.
4. **Detecção de Anomalias (Isolation Forest):** Identificação não supervisionada de eventos raros e pontos fora da curva, calculando um *anomaly score* para cada log.
5. **Correlação Topológica (NetworkX & PyVis):** Aplicação de cálculo de similaridade de cossenos entre os eventos anômalos isolados para gerar um grafo interativo, permitindo ao especialista visualizar as "famílias de erros" e rastrear a causa raiz.

## 🎯 Objetivos

1. **Automatizar** o processo de identificação de logs anômalos.
2. **Reduzir o MTTD e MTTI**, agilizando a resposta a incidentes de forma sistêmica.
3. **Demonstrar eficácia** na detecção de eventos operacionais complexos utilizando métodos matemáticos não supervisionados.
4. **Otimizar recursos**, diminuindo a fadiga de alertas e garantindo o cumprimento de **SLAs**.

## 📊 Dashboard de Observabilidade

O projeto conta com uma interface gráfica interativa desenvolvida em Streamlit que atua como o painel de controle do SRE:
* **Métricas de RCA em Tempo Real:** Acompanhamento do *Mean Time To Detect* (MTTD) e *Mean Time To Investigate* (MTTI).
* **Visualização Temporal:** Linha do tempo de *Decision Scores* separando comportamentos normais e anômalos.
* **Grafo de Semelhança:** Mapa topológico em rede (com física aplicada) mapeando as conexões entre diferentes alertas.
* **Validação Human-in-the-Loop:** Tabela de auditoria para que o especialista valide (True/False) os alertas críticos, auxiliando no cálculo de *Precision@K*.

## 🛠️ Tecnologias e Bibliotecas Base

* `scikit-learn`: Isolation Forest, TF-IDF, TruncatedSVD.
* `drain3`: Automated Log Parsing.
* `networkx` / `pyvis`: Teoria dos Grafos e Física de rede.
* `pandas` / `numpy`: Manipulação de dados.
* `streamlit` / `plotly`: Interface de usuário e visualizações interativas web.

---

## ⚙️ Como Rodar o Projeto

### Pré-requisitos
```bash
pip install -r requirements.txt

```

#### Faça o download do Docker Compose:

```bash
apt-get install docker-compose 
# ou
yum install docker-compose

```

### Inicialização do Sistema Base

O projeto possui um orquestrador central que sobe a interface web e mantém o motor analítico em background.

```bash
python main.py

```

> ⚠️ O painel do Streamlit será aberto automaticamente no seu navegador na porta 8501.

### Rodando em Ambiente Docker (Recomendado)

Para simular a infraestrutura localmente e gerar falhas:

1. Suba os containers da aplicação:

```bash
docker-compose -f docker/docker-compose.yml up -d --build 
# ou 
docker compose -f docker/docker-compose.yml up -d --force-recreate

```

2. Execução do docker controlada (Sobe por 10min e derruba):

```bash
docker compose -f docker/docker-compose.yml up -d ; sleep 10m ; docker compose -f docker/docker-compose.yml down

```

3. Injeção de Caos (Pumba) - Comandos para forçar erros e anomalias na infra:

```bash
# Derrubar a API abruptamente
docker exec pumba pumba kill --signal SIGKILL api

# Inserir atraso de rede (Delay de 3000ms)
docker exec pumba pumba netem --duration 30s delay --time 3000 api

```

4. Clean Up (Limpeza Geral):

```bash
cd docker ; docker-compose down --remove-orphans ; docker container prune -f; docker builder prune -f

```

### Rodando via Minikube (Kubernetes)

Para testes focados em orquestração de contêineres:

```bash
# 1. Inicie o motor do projeto
python main.py

# 2. Em outro terminal, inicie o cluster
cd minikube
./start-minikube.sh

# Para destruir o ambiente
./delete-minikube.sh

```

---

## 📬 Contato

guilherme.lopes13@hotmail.com

