# Detecção de Anomalias em Logs para Root Cause em Microsserviços

![AIOps](https://img.shields.io/badge/Focus-AIOps-blueviolet)
![Machine Learning](https://img.shields.io/badge/ML-Unsupervised-green)
![SRE](https://img.shields.io/badge/Area-SRE%20%26%20DevOps-orange)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

> TCC do MBA em IA e Big Data (ICMC-USP). Pipeline 100% open-source de
> detecção não supervisionada de anomalias em logs de aplicação, para
> acelerar Root Cause Analysis (RCA) em ambientes de microsserviços.

## 📋 Sobre o Projeto

Este projeto apresenta uma abordagem de **aprendizado não supervisionado**
aplicada a logs de aplicações em arquiteturas de microsserviços. O objetivo
é automatizar a identificação de eventos anômalos para acelerar a análise
de causa raiz (*Root Cause Analysis*), transformando a gestão de incidentes
de reativa para proativa, eliminando a dependência de regras estáticas.

## 🚀 Motivação e Contexto

Ambientes modernos de microsserviços oferecem resiliência e escalabilidade,
mas geram um volume de logs humanamente impossível de analisar manualmente
em tempo real.

* **Complexidade:** múltiplos serviços gerando logs simultâneos, sem um
  formato único.
* **Métricas em risco:** o tempo de análise manual eleva o **MTTD/MTTI**
  (tempo médio de detecção/investigação) e, por consequência, o **MTTR**.
* **Fadiga de alertas:** regras estáticas não acompanham a evolução do
  sistema e geram ruído.

A solução proposta usa técnicas de **AIOps** para detectar anomalias sem
depender de rótulos prévios, já que sistemas dinâmicos apresentam padrões
de falha em constante mudança.

## 🧠 Arquitetura do Pipeline Analítico

Fluxo implementado em `pipeline.py` (compartilhado entre o motor ao vivo e
os scripts de avaliação offline):

1. **Ingestão e parsing (Drain3):** extração automática de templates de log
   a partir de texto bruto semiestruturado, mascarando variáveis dinâmicas
   (IPs, hexadecimais, IDs).
2. **Vetorização (TF-IDF):** transforma os templates em matrizes esparsas,
   capturando frequência/relevância dos termos.
3. **Redução de dimensionalidade (`TruncatedSVD` ou `PCA`, configurável):**
   compressão da matriz esparsa em um espaço denso.
4. **Detecção de anomalias (`Isolation Forest` ou `One-Class SVM`,
   configurável):** cálculo de um *anomaly score* por evento de log.
5. **Agrupamento para RCA (`DBSCAN`):** agrupa os eventos anômalos
   detectados em "famílias de erro" (etapa opcional, `calcular_rca=True`).
6. **Correlação topológica (`NetworkX` + `PyVis`):** similaridade de
   cosseno entre eventos anômalos vira um grafo interativo, permitindo
   visualizar as famílias de erro e apoiar o RCA. *(Limitação atual: o
   grafo é descritivo/visual — ainda não calcula centralidade nem caminhos
   entre nós; ver seção de Observações.)*

> Escopo do TCC: pipeline simplificado, sem componentes como RAG/LLM —
> foco em clusterização clássica + correlação em grafo.

## 📊 Dashboard de Observabilidade

Interface em Streamlit (`modules/dashboard.py`) que funciona como painel de
SRE:

* Métricas de RCA em tempo real (MTTD/MTTI, `modules/mttd_mtti.py`).
* Linha do tempo de *decision scores*, separando normal vs. anômalo.
* Grafo de similaridade interativo (com física de rede).
* Tabela de validação *human-in-the-loop* (marcar alertas como
  verdadeiro/falso positivo) para apoiar o cálculo de Precision@K.
* Seleção, na própria tela, do algoritmo de detecção (`iForest`/`OCSVM`) e
  da técnica de redução (`PCA`/`SVD`) — grava a escolha em `config.json`.

## 🛠️ Tecnologias e Bibliotecas Base

* `drain3` — parsing automático de logs.
* `scikit-learn` — TF-IDF, `TruncatedSVD`/PCA, `IsolationForest`,
  `OneClassSVM`, `DBSCAN`.
* `networkx` / `pyvis` — grafo de correlação e física de rede.
* `pandas` / `numpy` — manipulação de dados (motor ao vivo e maioria dos
  experimentos).
* `polars` — usado nos scripts de avaliação sobre o BGL completo (ver
  Observações — necessário para não estourar memória).
* `streamlit` / `plotly` — dashboard e visualizações interativas.
* `matplotlib` / `seaborn` — gráficos estáticos usados nos experimentos.

---

## ⚙️ Pré-requisitos

* Python 3.10 ou 3.11.
* Git (para clonar o repositório e o submódulo de datasets).
* Docker + Docker Compose (opcional — só para o ambiente de simulação).
* Minikube + kubectl + Helm (opcional — só para o ambiente Kubernetes com
  Chaos Mesh; o script de setup instala o que faltar).

```bash
git clone --recurse-submodules <url-do-repositorio>
cd tcc_mba_ia_big_data

pip install -r requirements.txt
# Pacotes usados no código mas ausentes do requirements.txt no momento desta
# atualização — instale também (ver Observações):
pip install pyarrow polars
```

Se já clonou sem `--recurse-submodules`, baixe o submódulo de datasets
(`logpai`, espelho do [loghub](https://github.com/logpai/loghub)) com:

```bash
bash update_logdataset.sh
```

---

## ▶️ Como Rodar — Motor de Detecção ao Vivo (Dashboard)

O ponto de entrada atual é **`main_v6.py`** (não existe mais `main.py` na
raiz — veja Observações):

```bash
python main_v6.py
```

Isso sobe o dashboard Streamlit (porta **8501**, aberto automaticamente no
navegador) e, em paralelo, o motor de processamento em background, que
varre periodicamente as pastas de log configuradas. Use `Ctrl+C` no
terminal para encerrar os dois processos juntos.

O motor lê `config.json` a cada ciclo (gerado/editado pelo próprio
dashboard, mas também editável à mão):

```json
{"pastas": ["logs_filtrados"], "taxa_contaminacao": "auto", "algoritmo": "iforest", "reducao": "pca"}
```

* `pastas`: lista de diretórios de onde os `.log` são lidos. As opções
  "conhecidas" pelo dashboard ficam centralizadas em
  `modules/config_pastas.py` (`PASTAS_DISPONIVEIS`) — hoje configurado para
  `docker/logs_appficticio` e `minikube/k8s-chaos/logs_appficticio`.
* `taxa_contaminacao`: `"auto"` (estimada automaticamente, entre 0,5% e
  15%) ou um valor fixo (ex.: `0.05`).
* `algoritmo`: `"iforest"` ou `"ocsvm"`.
* `reducao`: `"pca"` ou `"svd"`.

Resultados de cada ciclo do motor ao vivo vão para `resultados/` (métricas
de MTTD/MTTI arquivadas por execução em
`resultados/historico_execucoes/metricas_rca_<timestamp>.json`, mais um
acumulado em `resultados/historico_metricas.json`).

---

## 🐳 Como Rodar — Ambiente Docker (gera logs fictícios + caos automático)

Sobe uma API Flask + worker + Redis + Nginx + coleta de log (Fluentd/
Logspout) e um container que injeta caos automaticamente:

```bash
# Build e subida dos containers
docker compose -f docker/docker-compose.yml up -d --build

# Acompanhar logs
docker compose -f docker/docker-compose.yml logs -f

# Derrubar tudo
docker compose -f docker/docker-compose.yml down --remove-orphans
```

O container `caos-aleatorio` já injeta caos sozinho, em loop (`docker/chaos.sh`):
a cada 30–90s (intervalo aleatório) sorteia um alvo (`api`, `redis` ou
`worker`) e um tipo de ataque (kill, delay de rede ou perda de pacotes) via
[Pumba](https://github.com/alexei-led/pumba). Não é preciso disparar nada
manualmente — mas se quiser forçar um ataque pontual:

```bash
# Matar a API abruptamente
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock gaiaadm/pumba:latest \
  kill --signal SIGKILL api

# Atraso de rede de 3s por 30s
docker run --rm --privileged -v /var/run/docker.sock:/var/run/docker.sock gaiaadm/pumba:latest \
  netem --duration 30s --tc-image gaiadocker/iproute2 delay --time 3000 api
```

> Não há mais um container chamado `pumba` de pé (esse era o modelo do
> `docker-compose copy.yml`, legado) — hoje o Pumba roda sob demanda via
> `docker run`, disparado pelo `caos-aleatorio`.

Os logs gerados caem em `docker/logs_appficticio/` (padronizado — ver
Observações), que é uma das pastas que o motor ao vivo já sabe ler.

Limpeza geral:

```bash
cd docker
docker compose down --remove-orphans
docker container prune -f
docker builder prune -f
```

---

## ☸️ Como Rodar — Ambiente Kubernetes (Minikube + Chaos Mesh)

Para testes com orquestração de contêineres de verdade (2 apps demo,
6 experimentos de Chaos Mesh sorteados aleatoriamente):

```bash
cd minikube
chmod +x start-minikube.sh
./start-minikube.sh
```

O script instala minikube/kubectl/helm se faltarem, sobe o cluster e, na
sequência, entra em `minikube/k8s-chaos/` para rodar o setup do Chaos Mesh
e os dois apps demo (`demo-app` e `chaos-log-app`) — detalhes de scripts em
`minikube/k8s-chaos/README.md`. Logs de ambos os apps vão para
`minikube/k8s-chaos/logs_appficticio/` (mesma convenção de nome usada no
Docker).

Para destruir o ambiente:

```bash
cd minikube
./delete-minikube.sh
```

---

## 🧪 Avaliação Offline / Reprodução dos Experimentos do TCC

Além do motor ao vivo, o repositório tem scripts de avaliação usados para
gerar os números e gráficos da monografia, todos operando sobre datasets
públicos do [LogHub](https://github.com/logpai/loghub) (HDFS e BGL).

### Datasets

`HDFS.log`, `BGL.log` e `anomaly_label.csv` **não estão versionados no
git** (estão no `.gitignore` — são arquivos grandes, o BGL completo tem
~745MB/4,7 milhões de linhas e o HDFS ~1,6GB). Formas de obtê-los:

* Submódulo `logpai` (`bash update_logdataset.sh`) — traz o loghub com
  amostras pequenas (ex.: `logpai/BGL/BGL_2k.log`) prontas para uso.
* Download automático via Hugging Face: `modules/dowloand_dataset_hugging.py`
  (usado como fallback pelos geradores de experimento HDFS quando o arquivo
  local não existe).
* Para os arquivos completos (BGL/HDFS "de verdade", tamanho real), baixe
  diretamente do LogHub/Zenodo e coloque em `experimento/HDFS.log`,
  `experimento/anomaly_label.csv` e `logpai/BGL/BGL.log` (caminhos default
  esperados pelos scripts abaixo).

### 1. Preparar o dataset balanceado do HDFS

```bash
python experimento/gerador_experimento_hdfsV3.py
```

Faz o parsing (Drain3) do `HDFS.log`, junta com os rótulos de
`anomaly_label.csv` e grava em `resultados/hdfs_treino_normal.parquet` e
`resultados/hdfs_teste_50_50.parquet` (base de teste balanceada 50/50
normal/anômalo). `gerador_experimento_hdfs.py`/`V2.py` são versões
anteriores, mantidas para referência.

### 2. Experimentos sobre o HDFS balanceado

```bash
python experimento/experimento_pipeline_iforest.py --anomaly-percentile 50
python experimento/experimento_pipeline_svm.py --amostra 33000
python experimento/experimento_pipeline_graficos.py --algoritmo iforest
python experimento/experimento_pipeline_graficosV2.py --algoritmo iforest --tag v2
python experimento/experimento_pipeline_literatura_v2.py
```

* `experimento_pipeline_iforest.py` / `experimento_pipeline_svm.py`:
  Isolation Forest / One-Class SVM **sem** redução de dimensionalidade,
  direto sobre o TF-IDF.
  `--amostra` no script do SVM limita o tamanho do treino (custo
  quadrático do SVM).
* `experimento_pipeline_graficos.py` / `graficosV2.py`: pipeline completo
  (com SVD) + geração dos gráficos/grafo usados na monografia;
  `--limite-amostra-ocsvm` controla o teto de linhas quando
  `--algoritmo ocsvm`.
* `experimento_pipeline_literatura.py` / `literatura_v2.py`: avaliação
  sobre a base rotulada "como está" (sem balanceamento artificial), usada
  para discutir generalização.

Cada execução grava em uma pasta nova e carimbada
`resultados/<AAAAMMDD_HHMMSS>_<tag>/` (ver `modules/run_output.py`), então
rodar o mesmo script várias vezes não sobrescreve resultados anteriores.

### 3. Walk-forward (TimeSeriesSplit) sobre HDFS/BGL

```bash
python avaliacao_walkforward.py --n-splits 5 --test-size 0.15 \
    --algoritmo iforest --reducao pca
```

Principais flags: `--pastas` (fontes de log), `--n-splits`, `--test-size`,
`--algoritmo {iforest,ocsvm}`, `--reducao {pca,svd}`, `--contaminacao`.

### 4. Avaliação sobre o BGL

```bash
# Versão pandas — recomendada só até algumas centenas de milhares de linhas
python experimento/BGL/avaliacao_bgl_v8.py --limite-linhas 200000

# Versão Polars — necessária para o BGL.log completo (4,7M linhas)
python experimento/BGL/avaliacao_bgl_v9_polaris.py
```

`avaliacao_bgl_v8.py` (pandas) é morto por *out-of-memory* ao tentar ler o
`BGL.log` completo em ambientes com ~8GB de RAM; use `--limite-linhas` para
uma amostra ou prefira a versão `v9_polaris.py` (Polars), que processa o
arquivo completo com uso de memória bem menor. `avaliacao_bgl.py` e
`avaliacao_bgl_v1..v7.py` são iterações anteriores, mantidas só como
histórico — não são a versão usada nos resultados finais do TCC.

### 5. Utilitário: consolidar templates do Drain3

```bash
cd drain3_states  # o script espera rodar de dentro dessa pasta
python ../get-drain-templates.py
```

Lê todos os `.bin` de estado do Drain3 em `drain3_states/` e consolida os
templates (com frequência) em `templates_drain_consolidados.json`.

---

## 🗂️ Estrutura do Repositório (resumo)

| Caminho | Conteúdo |
|---|---|
| `main_v6.py` | Ponto de entrada do motor ao vivo + dashboard (versão atual). |
| `pipeline.py` | Lógica de pré-processamento/treino/avaliação compartilhada entre motor ao vivo e avaliação offline. |
| `modules/` | Parsing (Drain3), pré-processamento, detector de anomalias, dashboard Streamlit, visualizações, métricas de MTTD/MTTI. |
| `docker/` | Ambiente Docker Compose (API fictícia + worker + Redis + coleta de log + injeção de caos com Pumba). |
| `minikube/` | Scripts para subir/derrubar o ambiente Kubernetes; `minikube/k8s-chaos/` tem os manifests do Chaos Mesh e os apps demo. |
| `experimento/` | Scripts de avaliação offline (HDFS/BGL) usados para os resultados do TCC; `experimento/BGL/` tem as variantes específicas para o dataset BGL. |
| `logpai/` | Submódulo git apontando para o [loghub](https://github.com/logpai/loghub) (datasets públicos de log). |
| `resultados/` | Saída de cada execução (motor ao vivo e experimentos), em pastas carimbadas por data/hora. |
| `drain3_states/` | Estados binários persistidos do Drain3 (um por fonte de log/execução). |
| `old_versions/` | Versões anteriores de `main.py`/`dashboard.py`, mantidas só como histórico — não usadas pelo fluxo atual. |
| `config.json` | Configuração ativa do motor ao vivo (pastas, algoritmo, redução, contaminação). |

---

## 📝 Observações Importantes

* **`main.py` não existe mais na raiz.** O ponto de entrada atual é
  `main_v6.py`; `main_pca.py`, `main_svd.py`, `main_v3.py`, `main_v4.py` e
  `main_v5.py` são versões anteriores/experimentais mantidas para
  referência, e `old_versions/main.py` é a versão original que deu nome ao
  comando no README antigo.
* **`requirements.txt` está incompleto** em relação ao que o código
  importa hoje: `pyarrow` (leitura/escrita de `.parquet`, usada em vários
  scripts e no dashboard) e `polars` (usado em `avaliacao_bgl_v9_polaris.py`
  e nos geradores de experimento HDFS) não estão listados — instale-os à
  parte (`pip install pyarrow polars`) até o `requirements.txt` ser
  atualizado.
* **Isolamento de estado do Drain3:** ao criar um novo script de avaliação
  que reprocessa a mesma fonte de log várias vezes na mesma sessão, use um
  `nome_fonte`/arquivo `.bin` de estado isolado e resete-o antes de cada
  execução (padrão já aplicado em `avaliacao_walkforward.py`,
  `avaliacao_bgl_v8.py` e `avaliacao_bgl_v9_polaris.py`). Sem isso, o
  Drain3 acumula contagens de execuções anteriores da mesma sessão e
  distorce as métricas por template.
* **Grafo de correlação (NetworkX/PyVis)** é hoje puramente descritivo —
  não calcula centralidade nem caminhos entre nós. Fica registrado como
  sugestão para trabalhos futuros.
* **`docker-compose copy.yml`** é um arquivo de backup/legado (modelo com
  um container `pumba` fixo) — o compose ativo é `docker/docker-compose.yml`.
* Datasets brutos (`*.log`, `*.csv`, `*.parquet`) e modelos (`*.pkl`,
  `*.joblib`) são intencionalmente ignorados pelo git (`.gitignore`) — não
  espere encontrá-los após um `git clone` limpo; veja a seção de datasets
  acima para obtê-los.

---

## 📬 Contato

guilherme.lopes13@hotmail.com
