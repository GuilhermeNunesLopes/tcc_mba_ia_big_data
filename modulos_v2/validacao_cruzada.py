"""
Validação cruzada temporal (TimeSeriesSplit) para o motor v7.

O PROBLEMA QUE ISSO RESOLVE
----------------------------
No motor v6 (main_v6.py -> pipeline.treinar_e_avaliar), cada lote de logs
de uma fonte é cortado UMA ÚNICA VEZ, cronologicamente, em 80% treino /
20% teste (train_test_split(..., shuffle=False)). Duas consequências:

  1) A taxa de contaminação automática (pipeline.estimar_contaminacao_automatica)
     é calculada a partir de UM único ajuste do Isolation Forest auxiliar,
     sobre UMA única janela de treino. Se aquele lote específico, por
     coincidência, teve um trecho mais ruidoso ou mais calmo que o normal,
     a taxa de contaminação (e portanto o limiar de alerta) herda esse
     "azar"/"sorte" do corte, mesmo que o comportamento típico da fonte
     seja outro.
  2) Quando existe rótulo real (ex.: uma pasta com dados do BGL), o F1
     reportado é o de UM ÚNICO corte — não há nenhuma evidência de que o
     resultado seria parecido se o corte tivesse caído um pouco antes ou
     um pouco depois na linha do tempo (a mesma pergunta que
     avaliacao_walkforward.py já responde, mas só OFFLINE, como relatório
     separado — nunca alimenta de volta o motor ao vivo).

Este módulo cobre os dois pontos, usando sklearn.model_selection.TimeSeriesSplit
(k dobras cronológicas, janela de treino sempre crescente, nunca vê o
futuro) DENTRO do próprio motor ao vivo:

  - Roda k dobras, e em cada uma reaproveita pipeline.treinar_e_avaliar()
    (o MESMO código já testado do v6 — nada de reimplementar engenharia de
    features aqui, para não arriscar divergir) com calcular_rca=False (o
    DBSCAN/silhueta não interessa nas dobras internas, só métricas).
  - Agrega, entre as dobras: a taxa de contaminação média (mais estável que
    a de uma dobra só) e, quando há rótulo, o F1 médio de cada algoritmo
    candidato — permitindo escolher automaticamente entre iForest e OCSVM
    pelo desempenho cruzado, em vez de depender só da escolha manual do
    operador no dashboard.
  - Sinaliza "degenerada=True" quando o resultado da CV não é confiável
    (poucas dobras, contaminação sempre no piso/teto, ou F1 médio muito
    baixo) — main_v7.py usa esse sinal para decidir se vale a pena
    confiar no modelo novo ou cair de volta no último modelo bom
    persistido (ver modulos_v2/model_registry.py).

POR QUE NÃO USAR TODOS OS FOLDS EM TODO BATCH
-----------------------------------------------
Os lotes do motor ao vivo (docker/logs_appficticio,
minikube/k8s-chaos/logs_appficticio) costumam ser pequenos — main_v6.py já
teria pulado o lote inteiro se tivesse menos de 20 logs. Com poucas
centenas de logs, 5 dobras cronológicas dariam janelas de validação
minúsculas e sem sentido estatístico. determinar_n_splits() calcula quantas
dobras cabem com um tamanho mínimo razoável por dobra e, se não der para
pelo menos 2 dobras, validar_com_time_series_split() retorna None — o
chamador (pipeline_v2.py) cai de volta no comportamento do v6 (uma
estimativa de contaminação só, sem seleção automática de algoritmo) para
aquele lote específico, sem quebrar nada.
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import pipeline as pipeline

# Tamanho mínimo de amostras por dobra para a CV fazer sentido estatístico.
# Abaixo disso, uma dobra de validação de poucas dezenas de logs dá uma
# estimativa de F1/contaminação instável demais para ser útil.
MIN_AMOSTRAS_POR_DOBRA = 60

# Teto de dobras — mais que isso não compensa o custo extra de treino
# repetido no motor ao vivo (cada dobra treina um Isolation Forest/OCSVM
# do zero).
MAX_DOBRAS = 5

# OneClassSVM (libsvm) tem custo pior que quadrático no número de amostras
# (achado validado nesta mesma sessão, com o HDFS/BGL completos) — incluir
# "ocsvm" como candidato em dobras de treino grandes tornaria a CV do motor
# ao vivo lenta demais. Acima deste tamanho de treino, a dobra pula OCSVM e
# só avalia iForest, mesmo se "ocsvm" estiver na lista de candidatos.
OCSVM_MAX_AMOSTRAS_TREINO_CV = 20000

# Limiares para marcar o resultado da CV como "degenerada" (não confiável):
F1_MINIMO_CONFIAVEL = 0.05
FRACAO_MAXIMA_DOBRAS_NO_LIMITE = 0.8  # se >=80% das dobras baterem no piso/teto de contaminação


def determinar_n_splits(n_amostras, min_amostras_por_dobra=MIN_AMOSTRAS_POR_DOBRA, max_dobras=MAX_DOBRAS):
    """Quantas dobras cronológicas cabem com um tamanho mínimo razoável por dobra."""
    n_dobras_possivel = (n_amostras // min_amostras_por_dobra) - 1
    return int(np.clip(n_dobras_possivel, 0, max_dobras))


def validar_com_time_series_split(df_fonte, reducao_ativa="pca", algoritmos_candidatos=("iforest",),
                                   n_splits=None, min_amostras_por_dobra=MIN_AMOSTRAS_POR_DOBRA,
                                   max_dobras=MAX_DOBRAS):
    """
    Roda TimeSeriesSplit sobre df_fonte (já ordenado cronologicamente) e
    retorna um relatório agregado, ou None se o lote for pequeno demais
    para uma CV com sentido estatístico (ver docstring do módulo).

    Retorno (dict):
      {
        "n_splits": int,
        "por_algoritmo": {
           "iforest": {"f1_media": .., "f1_desvio": .., "f1_por_dobra": [...],
                        "contaminacao_media": .., "contaminacao_por_dobra": [...],
                        "n_dobras_validas": int},
           "ocsvm": {...} ou ausente se nunca coube em nenhuma dobra,
        },
        "tem_rotulo": bool,
        "melhor_algoritmo": "iforest" | "ocsvm" | None (só quando tem_rotulo),
        "contaminacao_recomendada": float,
        "degenerada": bool,
        "motivo_degeneracao": str | None,
      }
    """
    n = len(df_fonte)
    if n_splits is None:
        n_splits = determinar_n_splits(n, min_amostras_por_dobra, max_dobras)

    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    acumulado = {algo: {"f1": [], "contaminacao": []} for algo in algoritmos_candidatos}
    tem_rotulo = False
    dobras_no_limite = 0
    dobras_validas_total = 0

    # Lê os limites direto de pipeline.py (CONTAMINACAO_AUTO_MINIMO/MAXIMO) em
    # vez de fixar um número aqui — se alguém ajustar esses limites em
    # pipeline.py, o diagnóstico de degeneração abaixo acompanha
    # automaticamente, sem precisar editar dois arquivos.
    limites_contaminacao = (
        getattr(pipeline, "CONTAMINACAO_AUTO_MINIMO", 0.005),
        getattr(pipeline, "CONTAMINACAO_AUTO_MAXIMO", 0.15),
    )

    for train_idx, val_idx in tscv.split(df_fonte):
        df_train_dobra = df_fonte.iloc[train_idx].reset_index(drop=True)
        df_val_dobra = df_fonte.iloc[val_idx].reset_index(drop=True)

        if len(df_train_dobra) < 20 or len(df_val_dobra) < 5:
            continue  # dobra pequena demais para o pipeline (mesmo piso de main_v6.py)

        dobra_teve_algum_algoritmo = False
        for algo in algoritmos_candidatos:
            if algo == "ocsvm" and len(df_train_dobra) > OCSVM_MAX_AMOSTRAS_TREINO_CV:
                continue

            try:
                resultado_dobra = pipeline.treinar_e_avaliar(
                    df_train_dobra, df_val_dobra,
                    taxa_contaminacao_ativa="auto",
                    algoritmo_ativo=algo,
                    reducao_ativa=reducao_ativa,
                    calcular_rca=False,
                )
            except Exception:
                # Uma dobra degenerada (ex.: vocabulário TF-IDF vazio num
                # trecho muito homogêneo) não deve derrubar a CV inteira —
                # só essa combinação dobra/algoritmo é descartada.
                continue

            dobra_teve_algum_algoritmo = True
            acumulado[algo]["contaminacao"].append(resultado_dobra["taxa_contaminacao_usada"])

            if resultado_dobra["metricas_ml"]:
                tem_rotulo = True
                acumulado[algo]["f1"].append(resultado_dobra["metricas_ml"]["F1_Score"])

            taxa = resultado_dobra["taxa_contaminacao_usada"]
            if limites_contaminacao and (taxa <= limites_contaminacao[0] or taxa >= limites_contaminacao[1]):
                dobras_no_limite += 1

        if dobra_teve_algum_algoritmo:
            dobras_validas_total += 1

    por_algoritmo = {}
    for algo, valores in acumulado.items():
        if not valores["contaminacao"]:
            continue  # este candidato nunca rodou em nenhuma dobra (ex.: OCSVM sempre grande demais)
        entrada = {
            "n_dobras_validas": len(valores["contaminacao"]),
            "contaminacao_media": float(np.mean(valores["contaminacao"])),
            "contaminacao_por_dobra": [float(v) for v in valores["contaminacao"]],
        }
        if valores["f1"]:
            entrada["f1_media"] = float(np.mean(valores["f1"]))
            entrada["f1_desvio"] = float(np.std(valores["f1"]))
            entrada["f1_por_dobra"] = [float(v) for v in valores["f1"]]
        por_algoritmo[algo] = entrada

    if not por_algoritmo:
        return None  # nenhuma dobra produziu resultado utilizável para nenhum candidato

    melhor_algoritmo = None
    if tem_rotulo:
        candidatos_com_f1 = {a: v["f1_media"] for a, v in por_algoritmo.items() if "f1_media" in v}
        if candidatos_com_f1:
            melhor_algoritmo = max(candidatos_com_f1, key=candidatos_com_f1.get)

    algo_para_contaminacao = melhor_algoritmo or next(iter(por_algoritmo))
    contaminacao_recomendada = por_algoritmo[algo_para_contaminacao]["contaminacao_media"]

    # ---- Diagnóstico de degeneração ----
    degenerada = False
    motivo_degeneracao = None

    if dobras_validas_total == 0:
        degenerada = True
        motivo_degeneracao = "nenhuma dobra produziu resultado válido"
    elif tem_rotulo and melhor_algoritmo is not None and por_algoritmo[melhor_algoritmo]["f1_media"] < F1_MINIMO_CONFIAVEL:
        degenerada = True
        motivo_degeneracao = (
            f"F1 médio da CV para '{melhor_algoritmo}' ficou em "
            f"{por_algoritmo[melhor_algoritmo]['f1_media']:.4f} (< {F1_MINIMO_CONFIAVEL}) — "
            "modelo não está separando anomalia de normal de forma confiável neste lote"
        )
    elif dobras_no_limite / max(dobras_validas_total, 1) >= FRACAO_MAXIMA_DOBRAS_NO_LIMITE:
        degenerada = True
        motivo_degeneracao = (
            f"{dobras_no_limite}/{dobras_validas_total} dobras bateram no piso/teto de contaminação "
            f"({limites_contaminacao[0]:.1%}/{limites_contaminacao[1]:.1%}) — "
            "lote provavelmente sem variação suficiente (quase tudo igual, ou quase tudo incomum)"
        )

    return {
        "n_splits": n_splits,
        "por_algoritmo": por_algoritmo,
        "tem_rotulo": tem_rotulo,
        "melhor_algoritmo": melhor_algoritmo,
        "contaminacao_recomendada": contaminacao_recomendada,
        "degenerada": degenerada,
        "motivo_degeneracao": motivo_degeneracao,
    }
