"""
pipeline_v2.py — camada de orquestração do motor v7.

O que este módulo FAZ:
  - Decide, para cada lote/fonte, se vale a pena rodar validação cruzada
    temporal (modulos_v2.validacao_cruzada) e, se sim, usa o resultado
    para (a) obter uma taxa de contaminação mais estável e, opcionalmente,
    (b) escolher automaticamente entre iForest/OCSVM pelo F1 cruzado.
  - Detecta quando esse resultado saiu degenerado (dados insuficientes,
    F1 baixo demais, contaminação sempre no piso/teto) e, nesse caso,
    tenta reaproveitar o último modelo bom persistido para aquela fonte
    (modulos_v2.model_registry) em vez de implantar um modelo novo ruim.
  - Quando treina de fato, persiste o modelo (+ vectorizer/redução/scaler)
    para poder servir de fallback em um lote futuro.

O que este módulo NÃO FAZ (de propósito):
  - Não reimplementa engenharia de features, treino ou inferência — tudo
    isso continua sendo feito pelas MESMAS funções de pipeline.py e
    modules/anomaly_detector.py já usadas pelo v6 (preparar_features,
    process_log_anomalies). Só a ORQUESTRAÇÃO ao redor delas é nova.

POR QUE UMA CÓPIA DE treinar_e_avaliar() EM VEZ DE CHAMAR pipeline.treinar_e_avaliar()
DIRETAMENTE
------------------------------------------------------------------------------------
pipeline.treinar_e_avaliar() já faz praticamente tudo que a gente precisa,
mas o dict que ela retorna não inclui o vectorizer/modelo de
redução/scaler AJUSTADOS no treino (só o vectorizer) — sem os três, não dá
para rodar inferência depois em cima de um modelo salvo (o fallback
descrito acima), porque um TF-IDF/PCA/SVD novo, ajustado em outro lote,
produziria um espaço de features diferente e incompatível com o modelo
salvo. Duas opções: (1) alterar a assinatura de retorno de
pipeline.treinar_e_avaliar() para incluir esses três objetos, ou (2)
duplicar aqui a orquestração (chamando as MESMAS pipeline.preparar_features
e anomaly_detector.process_log_anomalies, sem reescrever a lógica delas).
Optou-se por (2): pipeline.py é código já validado, citado na monografia
e usado por main_v6.py, avaliacao_walkforward.py e experimento_pipeline_*.py
— mudar sua assinatura de retorno é um risco desnecessário para uma
necessidade que só o v7 tem. O mesmo tipo de trade-off (duplicar uma
função pequena em vez de acoplar o motor ao vivo a um script de avaliação,
ou vice-versa) já existe hoje em extrair_rotulo_bgl(), duplicada entre
main_v6.py e avaliacao_walkforward.py com o mesmo raciocínio — ver o
comentário de lá.
"""
import numpy as np

import modules.anomaly_detector as anomaly_detector
import modules.explainability as explainability
import modules.preprocessor as preprocessor
import pipeline as pipeline
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from modulos_v2 import model_registry, validacao_cruzada


def _treinar_e_avaliar_capturando_feats(df_train, df_test, taxa_contaminacao_ativa,
                                         algoritmo_ativo, reducao_ativa, top_n_termos,
                                         calcular_rca):
    """
    Mesma lógica de pipeline.treinar_e_avaliar(), só que também devolve
    `feats` (vectorizer + red_model + scaler ajustados no treino), que
    pipeline.treinar_e_avaliar() não expõe. Ver docstring do módulo para o
    porquê da duplicação em vez de alterar pipeline.py.
    """
    feats = pipeline.preparar_features(df_train, df_test, reducao_ativa=reducao_ativa)
    X_train, X_test = feats["X_train"], feats["X_test"]
    tfidf_test, vectorizer = feats["tfidf_test"], feats["vectorizer"]

    detalhes_auto_contaminacao = None
    if taxa_contaminacao_ativa == "auto":
        taxa_contaminacao_ativa, detalhes_auto_contaminacao = pipeline.estimar_contaminacao_automatica(X_train)

    percentil_corte = taxa_contaminacao_ativa * 100 if isinstance(taxa_contaminacao_ativa, (int, float)) else 3

    _, modelo_treinado, _, _, threshold_treinado = anomaly_detector.process_log_anomalies(
        df_original=df_train,
        X_tfidf=X_train,
        contamination=taxa_contaminacao_ativa,
        anomaly_percentile=percentil_corte,
        model=None,
        algorithm=algoritmo_ativo,
    )

    y_verdadeiro = pipeline.extrair_rotulo(df_test)

    df_resultado, _, metricas_ml, _, _ = anomaly_detector.process_log_anomalies(
        df_original=df_test,
        X_tfidf=X_test,
        y_true=y_verdadeiro,
        model=modelo_treinado,
        best_threshold=threshold_treinado,
        anomaly_percentile=percentil_corte,
        algorithm=algoritmo_ativo,
    )

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

    if calcular_rca and mask_anomalias.sum() > 2:
        X_anomalias_denso = X_test[posicoes_anomalas]
        clusterizador = DBSCAN(eps=0.5, min_samples=4)
        labels_clusters = clusterizador.fit_predict(X_anomalias_denso)
        df_resultado.loc[mask_anomalias, 'cluster_id'] = labels_clusters
        if len(set(labels_clusters)) > 1 and len(set(labels_clusters) - {-1}) > 0:
            resultado["score_silhueta"] = float(silhouette_score(X_anomalias_denso, labels_clusters))

    return resultado, feats


def _rodar_inferencia_com_pacote_salvo(pacote, df_test, top_n_termos, calcular_rca):
    """
    Aplica um modelo JÁ TREINADO (persistido por um lote anterior, não
    degenerado) a um novo df_test, sem re-treinar nada. Usado quando a CV
    do lote atual saiu degenerada e existe um modelo bom anterior para
    esta fonte (ver treinar_e_avaliar_v2).
    """
    vectorizer = pacote["vectorizer"]
    red_model = pacote["red_model"]
    scaler = pacote["scaler"]
    reducao_ativa = pacote["reducao"]
    algoritmo = pacote["algoritmo"]

    tfidf_test, _ = preprocessor.tfidf_vectorize(df_test, vectorizer=vectorizer)
    if reducao_ativa.lower() == "svd":
        red_test, _ = preprocessor.apply_truncated_svd(tfidf_test, svd_model=red_model)
    else:
        red_test, _ = preprocessor.apply_pca(tfidf_test, pca_model=red_model)

    temp_test = scaler.transform(np.log1p(df_test[['time_delta', 'log_rate_5m']]))
    X_test = np.hstack((red_test, temp_test))

    y_verdadeiro = pipeline.extrair_rotulo(df_test)

    df_resultado, _, metricas_ml, _, _ = anomaly_detector.process_log_anomalies(
        df_original=df_test,
        X_tfidf=X_test,
        y_true=y_verdadeiro,
        model=pacote["modelo_treinado"],
        best_threshold=pacote["threshold_treinado"],
        algorithm=algoritmo,
    )

    mask_anomalias = (df_resultado['pred_is_anomaly'] == 1).values
    posicoes_anomalas = df_resultado.loc[mask_anomalias, '_row_pos'].values
    explicacoes_por_posicao = explainability.explicar_anomalias(
        tfidf_test, vectorizer, posicoes_anomalas, top_n=top_n_termos
    )
    df_resultado['Termos_Explicativos'] = df_resultado['_row_pos'].map(explicacoes_por_posicao)

    resultado = {
        "df_resultado": df_resultado,
        "metricas_ml": metricas_ml,
        "modelo_treinado": pacote["modelo_treinado"],
        "threshold_treinado": pacote["threshold_treinado"],
        "vectorizer": vectorizer,
        "y_verdadeiro": y_verdadeiro,
        "score_silhueta": None,
        "taxa_contaminacao_usada": pacote["taxa_contaminacao_usada"],
        "detalhes_contaminacao_automatica": None,
    }

    if calcular_rca and mask_anomalias.sum() > 2:
        X_anomalias_denso = X_test[posicoes_anomalas]
        clusterizador = DBSCAN(eps=0.5, min_samples=4)
        labels_clusters = clusterizador.fit_predict(X_anomalias_denso)
        df_resultado.loc[mask_anomalias, 'cluster_id'] = labels_clusters
        if len(set(labels_clusters)) > 1 and len(set(labels_clusters) - {-1}) > 0:
            resultado["score_silhueta"] = float(silhouette_score(X_anomalias_denso, labels_clusters))

    return resultado


def treinar_e_avaliar_v2(df_train, df_test, taxa_contaminacao_ativa="auto",
                          algoritmo_ativo="iforest", reducao_ativa="pca",
                          top_n_termos=5, calcular_rca=True, nome_fonte=None,
                          permitir_selecao_automatica_algoritmo=False):
    """
    Equivalente v7 de pipeline.treinar_e_avaliar(), com três acréscimos:

      1) validação cruzada temporal (TimeSeriesSplit) sobre df_train, para
         uma taxa de contaminação automática mais estável e (opcional)
         seleção automática de algoritmo;
      2) fallback para o último modelo bom persistido, se a CV do lote
         atual sair degenerada e `nome_fonte` for informado;
      3) persistência do modelo treinado (quando não veio do fallback e a
         CV não estava degenerada), para servir de fallback em lotes
         futuros.

    `nome_fonte` deve ser um identificador estável da fonte (ex.: o nome
    da pasta) — é a chave usada tanto para persistir quanto para recuperar
    o modelo salvo. Se None, os itens 2 e 3 ficam desativados e o
    comportamento equivale a rodar só a CV (item 1) por cima do pipeline
    do v6 — útil para chamar isso fora do contexto do motor ao vivo (ex.:
    um script de avaliação pontual).
    """
    candidatos = ["iforest", "ocsvm"] if permitir_selecao_automatica_algoritmo else [algoritmo_ativo]

    relatorio_cv = validacao_cruzada.validar_com_time_series_split(
        df_train, reducao_ativa=reducao_ativa, algoritmos_candidatos=candidatos
    )

    algoritmo_final = algoritmo_ativo
    taxa_final = taxa_contaminacao_ativa
    if relatorio_cv is not None:
        if permitir_selecao_automatica_algoritmo and relatorio_cv["melhor_algoritmo"]:
            algoritmo_final = relatorio_cv["melhor_algoritmo"]
        if taxa_contaminacao_ativa == "auto" and relatorio_cv["contaminacao_recomendada"] is not None:
            taxa_final = relatorio_cv["contaminacao_recomendada"]

    cv_degenerada = bool(relatorio_cv is not None and relatorio_cv["degenerada"])

    pacote_fallback = None
    if cv_degenerada and nome_fonte:
        pacote_fallback = model_registry.carregar_ultimo_modelo_bom(nome_fonte)

    if pacote_fallback is not None:
        resultado = _rodar_inferencia_com_pacote_salvo(pacote_fallback, df_test, top_n_termos, calcular_rca)
        resultado["modelo_fallback_usado"] = True
        resultado["motivo_fallback"] = relatorio_cv["motivo_degeneracao"]
        resultado["algoritmo_usado"] = pacote_fallback["algoritmo"]
    else:
        resultado, feats = _treinar_e_avaliar_capturando_feats(
            df_train, df_test,
            taxa_contaminacao_ativa=taxa_final,
            algoritmo_ativo=algoritmo_final,
            reducao_ativa=reducao_ativa,
            top_n_termos=top_n_termos,
            calcular_rca=calcular_rca,
        )
        resultado["modelo_fallback_usado"] = False
        resultado["motivo_fallback"] = relatorio_cv["motivo_degeneracao"] if cv_degenerada else None
        resultado["algoritmo_usado"] = algoritmo_final

        if nome_fonte and not cv_degenerada:
            pacote = {
                "nome_fonte": nome_fonte,
                "algoritmo": algoritmo_final,
                "reducao": reducao_ativa,
                "vectorizer": feats["vectorizer"],
                "red_model": feats["red_model"],
                "scaler": feats["scaler"],
                "modelo_treinado": resultado["modelo_treinado"],
                "threshold_treinado": resultado["threshold_treinado"],
                "taxa_contaminacao_usada": resultado["taxa_contaminacao_usada"],
                "relatorio_validacao_cruzada": relatorio_cv,
                "n_amostras_treino": len(df_train),
            }
            model_registry.salvar_modelo_versionado(nome_fonte, pacote)
        elif nome_fonte and cv_degenerada and pacote_fallback is None:
            # CV degenerada e não havia nenhum modelo anterior para cair de
            # volta — segue com o modelo novo (mesmo comportamento do v6:
            # melhor um modelo possivelmente fraco do que nenhum), mas NÃO
            # persiste, para não "contaminar" o histórico de modelos bons
            # com um treino que a própria CV sinalizou como não confiável.
            pass

    resultado["relatorio_validacao_cruzada"] = relatorio_cv
    return resultado
