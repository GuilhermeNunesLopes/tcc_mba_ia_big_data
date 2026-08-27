import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ==============================================================================
# CONFIGURAÇÃO DE ROTA PARA A PASTA 'MODULES' (UM NÍVEL ACIMA)
# ==============================================================================
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_pai = os.path.abspath(os.path.join(pasta_atual, '..'))
if pasta_pai not in sys.path:
    sys.path.insert(0, pasta_pai)

# ==============================================================================
# IMPORTAÇÃO DOS MÓDULOS DA ARQUITETURA
# ==============================================================================
from modules.anomaly_detector import process_log_anomalies
from modules.metrics import calculate_metrics
from modules.preprocessor import apply_truncated_svd
import modules.visualizer as visualizer
import modules.run_output as run_output

# Amostra máxima de treino quando algoritmo="ocsvm": o OneClassSVM do
# scikit-learn (libsvm) tem custo mais que quadrático no número de amostras —
# rodar sem limite no treino completo (541 mil linhas) pode levar horas (ver
# achado 3.5 da Validação de Resultados v2, mesmo problema já corrigido em
# experimento_pipeline_svm.py). Aplicado só quando algoritmo == "ocsvm";
# iforest treina no conjunto inteiro normalmente.
LIMITE_AMOSTRA_OCSVM = 20000


def executar_experimento_com_graficos(algoritmo="iforest", limite_amostra_ocsvm=LIMITE_AMOSTRA_OCSVM):
    print("="*60)
    print("🚀 INICIANDO EXPERIMENTO ALINHADO AO MAIN.PY (COM GRÁFICOS)")
    print("="*60)

    # Pasta nova por execução (data/hora + algoritmo no nome) — cada rodada
    # (iforest, ocsvm, ...) fica isolada, permitindo comparar ao longo do tempo.
    pasta_saida = run_output.criar_pasta_execucao(f"graficos_{algoritmo}")

    path_treino = 'resultados/hdfs_treino_normal.parquet'
    path_teste = 'resultados/hdfs_teste_50_50.parquet'

    if not os.path.exists(path_treino) or not os.path.exists(path_teste):
        print("⚠️ ERRO: Execute primeiro o script gerador_experimento_hdfs.py")
        return

    # 1. CARREGAMENTO DOS DADOS
    df_treino = pd.read_parquet(path_treino)
    df_teste = pd.read_parquet(path_teste)

    if algoritmo == "ocsvm" and len(df_treino) > limite_amostra_ocsvm:
        print(f"⚠️ OCSVM: amostrando treino de {len(df_treino)} para {limite_amostra_ocsvm} linhas "
              f"(evita treino inviável — ver LIMITE_AMOSTRA_OCSVM).")
        df_treino = df_treino.sample(n=limite_amostra_ocsvm, random_state=42).reset_index(drop=True)

    X_treino_raw = df_treino.drop(columns=['BlockId', 'Label'], errors='ignore')
    X_teste_raw = df_teste.drop(columns=['BlockId', 'Label'], errors='ignore')
    X_teste_raw = X_teste_raw.reindex(columns=X_treino_raw.columns, fill_value=0)

    # 2. VETORIZAÇÃO (TF-IDF)
    print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ Engenharia de Features: TF-IDF...")
    tfidf = TfidfTransformer()
    X_treino_tfidf = tfidf.fit_transform(X_treino_raw)
    X_teste_tfidf = tfidf.transform(X_teste_raw)

    # 3. REDUÇÃO DE DIMENSIONALIDADE (SVD) - Alinhado ao main.py
    print(f"[{time.strftime('%H:%M:%S')}] 📉 Aplicando TruncatedSVD (como no fluxo principal)...")
    X_treino_reduzido, svd_model = apply_truncated_svd(X_treino_tfidf, n_components=100)
    X_teste_reduzido, _ = apply_truncated_svd(X_teste_tfidf, svd_model=svd_model)

    # 4. TREINAMENTO (APENAS COM LOGS NORMAIS)
    print(f"[{time.strftime('%H:%M:%S')}] 🧠 Treinando modelo {algoritmo}...")
    _, modelo_treinado, _, _, threshold_treinado = process_log_anomalies(
        df_original=df_treino,
        X_tfidf=X_treino_reduzido,
        y_true=None,
        algorithm=algoritmo,
        contamination=0.01
    )

    # 5. TESTE E DETECÇÃO (NO DATASET 50/50)
    print(f"[{time.strftime('%H:%M:%S')}] 🎯 Aplicando inferência no dataset 50/50...")
    df_resultado, _, metricas_ml, _, _ = process_log_anomalies(
        df_original=df_teste,
        X_tfidf=X_teste_reduzido,
        y_true=df_teste['Label'],
        model=modelo_treinado,
        algorithm=algoritmo,
        anomaly_percentile=50
    )

    # 6. AGRUPAMENTO TOPOLÓGICO (DBSCAN) - Alinhado ao main.py
    mask_anomalias = (df_resultado['pred_is_anomaly'] == 1).values
    qtd_anomalias = mask_anomalias.sum()

    if qtd_anomalias > 2:
        print(f"[{time.strftime('%H:%M:%S')}] 🧩 Agrupando anomalias com DBSCAN...")
        X_anomalias_denso = X_teste_reduzido[mask_anomalias]
        clusterizador = DBSCAN(eps=0.5, min_samples=4)
        labels_clusters = clusterizador.fit_predict(X_anomalias_denso)

        df_resultado.loc[mask_anomalias, 'cluster_id'] = labels_clusters

        if len(set(labels_clusters)) > 1 and len(set(labels_clusters) - {-1}) > 0:
            score_silhueta = silhouette_score(X_anomalias_denso, labels_clusters)
            print(f"[{time.strftime('%H:%M:%S')}] 📐 Silhouette Score do DBSCAN: {score_silhueta:.4f}")

    # 7. AVALIAÇÃO DE DESEMPENHO E MÉTRICAS
    precision, recall, f1, cm = calculate_metrics(
        y_true=df_resultado['y_true_label'],
        y_pred=df_resultado['pred_is_anomaly']
    )

    print("\n" + "="*40)
    print("📊 RESULTADOS MATEMÁTICOS DO EXPERIMENTO")
    print("="*40)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # ==========================================================================
    # 8. GERAÇÃO DE VISUALIZAÇÕES E GRÁFICOS INTERATIVOS
    # ==========================================================================
    print(f"\n[{time.strftime('%H:%M:%S')}] 📈 Gerando evidências gráficas...")

    # Prepara uma coluna 'Template' textual para o visualizer.py entender o conteúdo do bloco
    event_cols = [c for c in df_resultado.columns if str(c).startswith('E')]
    def criar_pseudo_template(row):
        eventos = [col for col in event_cols if row[col] > 0]
        return "Bloco contém: " + " | ".join(eventos)

    df_resultado['Template'] = df_resultado.apply(criar_pseudo_template, axis=1)
    df_resultado['Timestamp'] = df_resultado.index
    # 8.1 - Gráfico de Distribuição
    fig_dist = visualizer.plot_anomaly_distribution_plotly(df_resultado)
    fig_dist.write_html(os.path.join(pasta_saida, "grafico_distribuicao_iforestv1.html"))

    # 8.2 - Gráfico de Linha do Tempo
    fig_timeline = visualizer.plot_anomaly_timeline_plotly(df_resultado)
    fig_timeline.write_html(os.path.join(pasta_saida, "grafico_timeline_iforestv1.html"))

    # 8.3 - Matriz de Confusão
    fig_cm = visualizer.plot_confusion_matrix_plotly(cm)
    fig_cm.write_html(os.path.join(pasta_saida, "grafico_matriz_confusao_iforestv1.html"))

    # 8.4 - Grafo de Similaridade (NetworkX + PyVis)
    caminho_grafo = os.path.join(pasta_saida, "grafo_semelhanca_logs.html")
    visualizer.generate_interactive_network(df_resultado, output_path=caminho_grafo)

    # 8.5 - Novo Gráfico: Comparativo Antes e Depois
    fig_comparativo = visualizer.plot_comparativo_antes_depois(df_resultado)
    fig_comparativo.write_html(os.path.join(pasta_saida, "grafico_comparativo_iforestv1.html"))

    # 8.6 - Novo Gráfico: Métricas em Destaque
    fig_metricas = visualizer.plot_metricas_destaque(precision, recall, f1)
    fig_metricas.write_html(os.path.join(pasta_saida, "grafico_metricas_iforestv1.html"))

    print(f"[{time.strftime('%H:%M:%S')}] ✅ Todos os gráficos foram salvos em '{pasta_saida}'!")
    print("-> Dê um duplo clique nos arquivos .html gerados para visualizá-los diretamente no seu navegador.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento com gráficos (HDFS balanceado 50/50).")
    parser.add_argument("--algoritmo", type=str, default="iforest", choices=["iforest", "ocsvm"],
                         help="Algoritmo de detecção (padrão: iforest — ocsvm é bem mais lento).")
    parser.add_argument("--limite-amostra-ocsvm", type=int, default=LIMITE_AMOSTRA_OCSVM,
                         help=f"Tamanho máximo do treino quando --algoritmo ocsvm (padrão: {LIMITE_AMOSTRA_OCSVM}).")
    args = parser.parse_args()

    executar_experimento_com_graficos(algoritmo=args.algoritmo, limite_amostra_ocsvm=args.limite_amostra_ocsvm)
