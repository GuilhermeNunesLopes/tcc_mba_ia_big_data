"""
Lógica compartilhada de pré-processamento, treino e avaliação do detector
de anomalias em logs. Fatorado para fora de main_v4.py para que o mesmo
código sirva tanto para o pipeline ao vivo (um split por lote) quanto para
a avaliação walk-forward (múltiplos splits cronológicos), garantindo que
as duas rodagens usem exatamente a mesma lógica — nada de reimplementar
(e arriscar divergir) a engenharia de features em dois lugares.
"""

import numpy as np
import pandas as pd

import modules.preprocessor as preprocessor
import modules.anomaly_detector as anomaly_detector
import modules.explainability as explainability
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

# Faixa operacional em que a taxa de contaminação calculada automaticamente
# é travada (ver estimar_contaminacao_automatica). Sem isso, um lote quase
# sem variação de score (MAD≈0) ou já dominado por eventos incomuns
# devolveria 0% ou 100%, o que quebraria o treino/corte de classificação
# a jusante.
CONTAMINACAO_AUTO_MINIMO = 0.005
CONTAMINACAO_AUTO_MAXIMO = 0.15
CONTAMINACAO_AUTO_Z_THRESHOLD = 3.5

COLUNAS_POSSIVEIS_LABEL = ['Label', 'label', 'Anomaly', 'anomaly', 'Is_Anomaly', 'y_true']

TERMOS_WHITELIST = [
    "kube-proxy", "healthcheck", "get /healthz", "ping",
    "connection closed by peer", "liveness probe", "readiness probe"
]


def extrair_rotulo(df):
    """
    Procura uma coluna de rótulo conhecida no DataFrame e retorna um array
    binário. Retorna None se não houver coluna de rótulo (modo puramente
    não supervisionado). Usada SOMENTE para medir desempenho depois do
    fato — nunca para treinar ou calibrar threshold.
    """
    coluna_alvo = next((col for col in COLUNAS_POSSIVEIS_LABEL if col in df.columns), None)
    if coluna_alvo is None:
        return None
    return df[coluna_alvo].apply(
        lambda x: 1 if str(x).strip().lower() in ['anomaly', '1', 'true', 'anômalo', 'fail'] else 0
    ).values


def estimar_contaminacao_automatica(X_train, random_state=42, z_threshold=CONTAMINACAO_AUTO_Z_THRESHOLD,
                                     minimo=CONTAMINACAO_AUTO_MINIMO, maximo=CONTAMINACAO_AUTO_MAXIMO):
    """
    Estima a taxa de contaminação (fração esperada de anomalias) direto da
    distribuição de anomaly_score do próprio lote de treino — sem que um
    humano precise digitar um percentual no dashboard (era o slider
    "Alerta (%)", 0,1%-10%, valor fixo por padrão em 3,0%).

    Método: fit de um Isolation Forest neutro (contamination="auto") só
    para pontuar via score_samples() — que NÃO depende do parâmetro
    contamination (só o corte de classificação depende disso, não o
    ranking). Em cima desses scores, aplica-se o Modified Z-Score de
    Iglewicz & Hoaglin (1993), baseado na Median Absolute Deviation (MAD):

        M_i = 0.6745 * (score_i - mediana) / MAD

    MAD é robusto (os próprios outliers não distorcem mediana/MAD tanto
    quanto distorceriam média/desvio padrão). z_threshold=3.5 é o valor
    recomendado por Iglewicz & Hoaglin. A fração de logs do treino com
    M_i < -z_threshold vira a taxa de contaminação, travada em
    [minimo, maximo] para evitar degenerar (MAD≈0, lote quase todo
    homogêneo, etc.).

    Retorna (taxa_final: float, detalhes: dict) — detalhes é guardado para
    auditoria/explicabilidade (ver metricas_rca.json e o KPI do dashboard).
    """
    modelo_auxiliar = IsolationForest(n_estimators=200, contamination="auto",
                                       random_state=random_state, n_jobs=-1)
    modelo_auxiliar.fit(X_train)
    scores = modelo_auxiliar.score_samples(X_train)

    mediana = np.median(scores)
    mad = np.median(np.abs(scores - mediana))

    if mad == 0:
        return minimo, {
            "metodo": "fallback_mad_zero",
            "motivo": "MAD dos scores é zero (lote sem variação suficiente) — usado o piso mínimo",
            "mediana_score": float(mediana),
            "mad_score": 0.0,
            "z_threshold": z_threshold,
            "limites": [minimo, maximo],
            "n_amostras_treino": int(len(scores)),
        }

    z_scores = 0.6745 * (scores - mediana) / mad
    fracao_atipicos = float((z_scores < -z_threshold).mean())
    taxa_final = float(np.clip(fracao_atipicos, minimo, maximo))

    detalhes = {
        "metodo": "modified_z_score_mad",
        "z_threshold": z_threshold,
        "mediana_score": float(mediana),
        "mad_score": float(mad),
        "fracao_atipicos_bruta": fracao_atipicos,
        "taxa_final_apos_limites": taxa_final,
        "limites": [minimo, maximo],
        "n_amostras_treino": int(len(scores)),
    }
    return taxa_final, detalhes


def preprocessar_logs_brutos(df_logs):
    """
    Recebe o DataFrame cru (saída do Drain3, já concatenado de todas as
    pastas) e aplica: derivação de colunas Event/Source/Level, whitelist
    de ruído operacional, engenharia de features temporais e ordenação
    cronológica final. Não usa rótulo em nenhum passo.
    """
    df_logs = df_logs.copy()

    if 'Template' in df_logs.columns and 'Event' not in df_logs.columns:
        df_logs['Event'] = df_logs['Template']
        df_logs['Source'] = df_logs['Source_Folder']
        df_logs['Level'] = "INFO"

    # Whitelist (filtro de ruído operacional conhecido)
    mask_ignorar = pd.Series(False, index=df_logs.index)
    for termo in TERMOS_WHITELIST:
        if 'Raw_Log' in df_logs.columns:
            mask_ignorar = mask_ignorar | df_logs['Raw_Log'].str.contains(termo, case=False, na=False, regex=False)
        elif 'Event' in df_logs.columns:
            mask_ignorar = mask_ignorar | df_logs['Event'].str.contains(termo, case=False, na=False, regex=False)
    df_logs = df_logs[~mask_ignorar].copy()

    if df_logs.empty:
        return df_logs

    # Features temporais
    df_logs['Timestamp'] = pd.to_datetime(df_logs['Timestamp'], errors='coerce')
    df_logs = df_logs.dropna(subset=['Timestamp'])
    df_logs = df_logs.sort_values(by=['Source_Folder', 'Timestamp']).reset_index(drop=True)
    df_logs['time_delta'] = df_logs.groupby('Source_Folder')['Timestamp'].diff().dt.total_seconds().fillna(0)

    temp_indexed = df_logs.set_index('Timestamp')
    df_logs['log_rate_5m'] = temp_indexed.groupby('Source_Folder')['Raw_Log'].rolling('5min').count().values

    # Ordenação cronológica final (necessária para qualquer split temporal)
    df_logs = df_logs.sort_values(by='Timestamp').reset_index(drop=True)

    return df_logs


def preparar_features(df_train, df_test, reducao_ativa="pca", n_components=100):
    """
    Ajusta TF-IDF + PCA/SVD + StandardScaler SOMENTE no treino, e aplica
    (transform) no teste. Retorna tudo que é preciso tanto para treinar
    quanto para depois explicar/agrupar as anomalias do teste.
    """
    tfidf_train, vectorizer = preprocessor.tfidf_vectorize(df_train, vectorizer=None)

    if reducao_ativa.lower() == "svd":
        red_train, red_model = preprocessor.apply_truncated_svd(tfidf_train, svd_model=None, n_components=n_components)
    else:
        red_train, red_model = preprocessor.apply_pca(tfidf_train, pca_model=None, n_components=n_components)

    scaler = StandardScaler()
    temp_train = scaler.fit_transform(np.log1p(df_train[['time_delta', 'log_rate_5m']]))
    X_train = np.hstack((red_train, temp_train))

    tfidf_test, _ = preprocessor.tfidf_vectorize(df_test, vectorizer=vectorizer)
    if reducao_ativa.lower() == "svd":
        red_test, _ = preprocessor.apply_truncated_svd(tfidf_test, svd_model=red_model)
    else:
        red_test, _ = preprocessor.apply_pca(tfidf_test, pca_model=red_model)

    temp_test = scaler.transform(np.log1p(df_test[['time_delta', 'log_rate_5m']]))
    X_test = np.hstack((red_test, temp_test))

    return {
        "X_train": X_train, "X_test": X_test,
        "tfidf_train": tfidf_train, "tfidf_test": tfidf_test,
        "vectorizer": vectorizer, "red_model": red_model, "scaler": scaler,
    }


def treinar_e_avaliar(df_train, df_test, taxa_contaminacao_ativa="auto",
                       algoritmo_ativo="iforest", reducao_ativa="pca",
                       top_n_termos=5, calcular_rca=True):
    """
    Roda o pipeline completo (features -> treino não supervisionado ->
    inferência -> explicabilidade -> [opcional] agrupamento RCA) para UM
    par (df_train, df_test) e retorna um dicionário com tudo que interessa.

    `calcular_rca=False` pula o DBSCAN/silhouette (usado na avaliação
    walk-forward, onde só as métricas de detecção importam e rodar RCA em
    dezenas de splits seria custo desnecessário).
    """
    feats = preparar_features(df_train, df_test, reducao_ativa=reducao_ativa)
    X_train, X_test = feats["X_train"], feats["X_test"]
    tfidf_test, vectorizer = feats["tfidf_test"], feats["vectorizer"]

    # ---- FASE 1: TREINO (100% não supervisionado) ----
    detalhes_auto_contaminacao = None
    if taxa_contaminacao_ativa == "auto":
        taxa_contaminacao_ativa, detalhes_auto_contaminacao = estimar_contaminacao_automatica(X_train)
        print(f"🧮 Taxa de contaminação calculada automaticamente: {taxa_contaminacao_ativa:.4%} "
              f"(método: Modified Z-Score/MAD, threshold={detalhes_auto_contaminacao['z_threshold']}, "
              f"{detalhes_auto_contaminacao['n_amostras_treino']} logs de treino)")

    percentil_corte = taxa_contaminacao_ativa * 100 if isinstance(taxa_contaminacao_ativa, (int, float)) else 3

    _, modelo_treinado, _, _, threshold_treinado = anomaly_detector.process_log_anomalies(
        df_original=df_train,
        X_tfidf=X_train,
        contamination=taxa_contaminacao_ativa,
        anomaly_percentile=percentil_corte,
        model=None,
        algorithm=algoritmo_ativo
    )

    # ---- FASE 2: INFERÊNCIA (threshold congelado do treino) ----
    # BUG CORRIGIDO: esta chamada não repassava anomaly_percentile=percentil_corte.
    # Quando não há rótulo (modo ao vivo, y_true=None em ambas as fases),
    # threshold_treinado também é None (o grid search de optimize_isolation_forest
    # só roda com rótulo disponível) — então process_log_anomalies cai no ramo
    # `else: limiar_estatistico = np.percentile(decision_scores, anomaly_percentile)`,
    # e sem o argumento aqui, `anomaly_percentile` usava o valor padrão da
    # assinatura da função (3), IGNORANDO taxa_contaminacao_ativa por completo —
    # fosse ela o valor manual do slider ou, agora, o valor calculado
    # automaticamente. Isso ficava mascarado porque o padrão antigo do slider
    # (3,0%) coincidia com esse "3" hardcoded; qualquer valor diferente de 3%
    # nunca chegava a influenciar a classificação real.
    y_verdadeiro = extrair_rotulo(df_test)  # só para medir, nunca para decidir

    df_resultado, _, metricas_ml, _, _ = anomaly_detector.process_log_anomalies(
        df_original=df_test,
        X_tfidf=X_test,
        y_true=y_verdadeiro,
        model=modelo_treinado,
        best_threshold=threshold_treinado,
        anomaly_percentile=percentil_corte,
        algorithm=algoritmo_ativo
    )

    # ---- EXPLICABILIDADE POR TERMOS ----
    # df_resultado já pode estar reordenado por anomaly_score; '_row_pos'
    # guarda a posição original, que é a mesma ordem de tfidf_test.
    mask_anomalias = (df_resultado['pred_is_anomaly'] == 1).values
    posicoes_anomalas = df_resultado.loc[mask_anomalias, '_row_pos'].values

    explicacoes_por_posicao = explainability.explicar_anomalias(
        tfidf_test, vectorizer, posicoes_anomalas, top_n=top_n_termos
    )
    df_resultado['Termos_Explicativos'] = df_resultado['_row_pos'].map(explicacoes_por_posicao)

    resultado = {
        "df_resultado": df_resultado,
        "metricas_ml": metricas_ml,
        "modelo_treinado": modelo_treinado,
        "threshold_treinado": threshold_treinado,
        "vectorizer": vectorizer,
        "y_verdadeiro": y_verdadeiro,
        "score_silhueta": None,
        "taxa_contaminacao_usada": taxa_contaminacao_ativa,
        "detalhes_contaminacao_automatica": detalhes_auto_contaminacao,
    }

    # ---- AGRUPAMENTO RCA (DBSCAN) — opcional ----
    if calcular_rca and mask_anomalias.sum() > 2:
        # Usa '_row_pos' (posição original) para buscar os vetores certos em
        # X_test, já que df_resultado pode estar fora de ordem posicional.
        X_anomalias_denso = X_test[posicoes_anomalas]
        clusterizador = DBSCAN(eps=0.5, min_samples=4)
        labels_clusters = clusterizador.fit_predict(X_anomalias_denso)

        df_resultado.loc[mask_anomalias, 'cluster_id'] = labels_clusters

        if len(set(labels_clusters)) > 1 and len(set(labels_clusters) - {-1}) > 0:
            resultado["score_silhueta"] = float(silhouette_score(X_anomalias_denso, labels_clusters))

    return resultado
