# Detecção de Anomalias em Logs para Root Cause em Microsserviços

![AIOps](https://img.shields.io/badge/Focus-AIOps-blueviolet)
![Machine Learning](https://img.shields.io/badge/ML-Unsupervised-green)
![SRE](https://img.shields.io/badge/Area-SRE%20%26%20DevOps-orange)

## 📋 Sobre o Projeto

Este projeto apresenta uma abordagem de **Aprendizado Não Supervisionado** aplicada a logs de aplicações em arquiteturas de microsserviços. O objetivo é automatizar a identificação de eventos anômalos para acelerar a análise de causa raiz (*Root Cause Analysis*), transformando a gestão de incidentes de reativa para proativa.

## 🚀 Motivação e Contexto

Ambientes modernos de microsserviços oferecem resiliência e escalabilidade, mas geram um volume de dados (logs) humanamente impossível de analisar manualmente em tempo real. 

### O Problema
* **Complexidade:** Milhares de sub-ambientes gerando logs simultâneos.
* **Métricas em Risco:** O aumento no tempo de análise manual eleva o **MTTI** (Tempo Médio de Identificação) e o **MTTR** (Tempo Médio de Resolução).
* **Fadiga de Alertas:** Equipes de SRE e DevOps sobrecarregadas por notificações excessivas ou regras estáticas que não acompanham a evolução do sistema.

### A Solução
Utilizar técnicas de **AIOps** para detectar anomalias sem a necessidade de rótulos prévios. Como sistemas dinâmicos apresentam padrões de falha em constante evolução, o aprendizado não supervisionado permite identificar comportamentos incomuns que antecedem falhas críticas.

## 🎯 Objetivos

1.  **Automatizar** o processo de identificação de logs anômalos.
2.  **Reduzir o MTTI**, agilizando a resposta a incidentes.
3.  **Demonstrar eficácia** na detecção de eventos operacionais complexos em sistemas distribuídos.
4.  **Otimizar recursos**, diminuindo a fadiga de alertas e garantindo o cumprimento de **SLAs**.

## 🛠️ Tecnologias e Abordagens

* **Arquitetura:** Microsserviços e Sistemas Distribuídos.
* **Inteligência Artificial:** Aprendizado Não Supervisionado (AIOps).
* **Foco Setorial:** Aplicações de alta criticidade (Bancos, Credores e Fintechs).

## 📊 Benefícios Esperados

| Métrica | Impacto Esperado |
| :--- | :--- |
| **MTTI** | Redução significativa através da detecção automatizada. |
| **SLA** | Maior conformidade devido à rapidez na identificação de falhas. |
| **Resiliência** | Identificação proativa de comportamentos que antecedem incidentes. |
| **Operação** | Menor carga cognitiva para os times de engenharia (SRE/DevOps). |

---
*Este estudo visa conferir vantagem competitiva através da alta disponibilidade e resiliência de aplicações de ponta a ponta.*

## Como Rodar ? 
### Docker:
1. Execute:
```bash
 pip install -r requeriment.txt
 ```
2. Faça o dowloand do Docker Compose:
```bash
apt get install docker-compose 
yum install docker-compose
```
3. Suba os containers da aplicação:
```bash
docker-compose -f docker/docker-compose.yml up -d --build
```
4. Caso precise recriar os containers: 
```bash
docker-compose up -d --build --force-recreate
```
### Start anomaly detection script:
1. Execute: 
```bash
python3 main.py
```
2. Informe o token do hungging-face
3. Espere a execução.

## Comandos do pumba para forçar erros / anomalias
```bash
docker exec pumba pumba kill --signal SIGKILL api
```
```bash
docker exec pumba pumba netem --duration 30s delay --time 3000 api
```
## Clean Up containers

### Para todos os containers e remove redes órfãs
```bash
cd docker ; docker-compose down --remove-orphans ; docker container prune -f; docker builder prune -f
```
