import sys
import os
import time
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

# ==============================================================================
# CONFIGURAÇÃO DE ROTA PARA A PASTA 'MODULES'
# ==============================================================================
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_pai = os.path.abspath(os.path.join(pasta_atual, '..'))
if pasta_pai not in sys.path:
    sys.path.insert(0, pasta_pai)

# ==============================================================================
# IMPORTAÇÃO DOS MÓDULOS DA ARQUITETURA
# ==============================================================================
from modules.anomaly_detector import process_log_anomalies
from modules.preprocessor import apply_truncated_svd
from modules.parse_system import automatic_drain_parse
import modules.visualizer as visualizer

def executar_experimento_completo_literatura():
    print("="*60)
    print("🚀 INICIANDO EXPERIMENTO COMPLETO - LITERATURA (100% DOS DADOS HDFS)")
    print("="*60)
    
    log_path = 'HDFS_42k.log'
    #log_path = 'HDFS_23k.log'
    #log_path = 'HDFS.log'
    label_path = 'anomaly_label.csv'
    output_dir = 'resultados'
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    if not os.path.exists(log_path) or not os.path.exists(label_path):
        print(f"⚠️ ERRO: '{log_path}' ou '{label_path}' não encontrados na pasta atual.")
        return

    # ---------------------------------------------------------
    # 1. PARSING E AGRUPAMENTO ESCALÁVEL (PARA ARQUIVOS GIGANTES)
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] [1/4] Extraindo templates e agrupando blocos em lotes...")
    
    batch_log = automatic_drain_parse(log_path, nome_fonte="HDFS_Completo", tamanho_lote=100000)
    
    lista_frequencias_lotes = []

    # O "for" volta para garantir que a RAM não estoure
    for i, df_lote in enumerate(batch_log):
        # 1. Extrai BlockId e limpa
        df_lote['BlockId'] = df_lote['Raw_Log'].str.extract(r'(blk_-?\d+)')
        df_lote = df_lote.dropna(subset=['BlockId'])
        df_lote = df_lote.rename(columns={'Cluster_ID': 'EventId'})
        
        # 2. Conta as ocorrências dos eventos APENAS neste lote (MUITO leve na RAM)
        # O reset_index com name='Count' cria uma tabela com: BlockId | EventId | Count
        freq_lote = df_lote.groupby(['BlockId', 'EventId']).size().reset_index(name='Count')
        
        # 3. Guarda a contagem numérica e a função descarta o texto pesado do lote (Raw_Log)
        lista_frequencias_lotes.append(freq_lote)
        print(f"   -> Lote {i+1} agregado e texto descartado da RAM.")

    # 4. Fora do loop: Junta apenas as matrizes numéricas e soma os blocos que 
    # eventualmente começaram em um lote e terminaram em outro.
    print(f"\n[{time.strftime('%H:%M:%S')}] Consolidando frequências globais...")
    df_frequencias_totais = pd.concat(lista_frequencias_lotes, ignore_index=True)
    
    # Soma as contagens e cria a matriz final (BlockId nas linhas, EventId nas colunas)
    df_grouped = df_frequencias_totais.groupby(['BlockId', 'EventId'])['Count'].sum().unstack(fill_value=0).reset_index()
    df_grouped.columns = ['BlockId'] + [f'E{col}' for col in df_grouped.columns if col != 'BlockId']

    print(f"   -> Matriz estruturada final com {len(df_grouped)} blocos totais.")

    # ---------------------------------------------------------
    # 2. APLICAÇÃO DO GROUND TRUTH
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] [2/4] Cruzando dados reais com o anomaly_label.csv...")
    labels = pd.read_csv(label_path)
    labels['Label'] = labels['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
    
    df_final = pd.merge(df_grouped, labels, on='BlockId', how='inner')
    X_raw = df_final.drop(columns=['BlockId', 'Label'])
    y_true = df_final['Label']

    taxa_real_anomalia = (y_true.sum() / len(y_true))
    print(f"   -> Total avaliado: {len(df_final)} blocos.")
    print(f"   -> Anomalias reais detectadas no label: {y_true.sum()} ({taxa_real_anomalia:.2%})")

    # ---------------------------------------------------------
    # 3. PRÉ-PROCESSAMENTO (TF-IDF + SVD)
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] [3/4] Aplicando vetorização e compressão SVD...")
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_raw)
    X_reduzido, svd_model = apply_truncated_svd(X_tfidf, n_components=100)

    # ---------------------------------------------------------
    # 4. TREINAMENTO E INFERÊNCIA DO MODELO
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] [4/4] Executando detecção via Isolation Forest...")
    
    # O process_log_anomalies otimiza internamente com F1-Score usando 20% para validação cruzada
    # Passamos os labels originais para que ele tente encontrar o melhor threshold sem vazar dados.
    df_resultado, modelo_treinado, metricas, best_params, threshold = process_log_anomalies(
        df_original=df_final, 
        X_tfidf=X_reduzido, 
        y_true=y_true, 
        algorithm="iforest"
    )

    # ---------------------------------------------------------
    # 5. CÁLCULO DAS MÉTRICAS ACADÊMICAS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📊 RESULTADOS ACADÊMICOS DO EXPERIMENTO (100% BASE)")
    print("="*50)
    
    # Recalculando localmente para garantir o print explícito
    prec = precision_score(y_true, df_resultado['pred_is_anomaly'], zero_division=0)
    rec = recall_score(y_true, df_resultado['pred_is_anomaly'], zero_division=0)
    f1 = f1_score(y_true, df_resultado['pred_is_anomaly'], zero_division=0)
    
    # PR-AUC
    scores_decision = -modelo_treinado.decision_function(X_reduzido)
    precisions_curve, recalls_curve, _ = precision_recall_curve(y_true, scores_decision)
    pr_auc = auc(recalls_curve, precisions_curve)

    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    # ==========================================================================
    # 6. GERAÇÃO DE EVIDÊNCIAS GRÁFICAS
    # ==========================================================================
    print(f"\n[{time.strftime('%H:%M:%S')}] 📈 Gerando evidências visuais...")
    
    df_resultado['Timestamp'] = df_resultado.index
    
    event_cols = [c for c in df_resultado.columns if str(c).startswith('E')]
    def criar_pseudo_template(row):
        eventos = [col for col in event_cols if row[col] > 0]
        return "Bloco: " + " | ".join(eventos)
    df_resultado['Template'] = df_resultado.apply(criar_pseudo_template, axis=1)

    fig_dist = visualizer.plot_anomaly_distribution_plotly(df_resultado)
    fig_dist.write_html("resultados/grafico_distribuicao_completo.html")
    
    fig_timeline = visualizer.plot_anomaly_timeline_plotly(df_resultado)
    fig_timeline.write_html("resultados/grafico_timeline_completo.html")

    print(f"[{time.strftime('%H:%M:%S')}] ✅ Execução finalizada! Gráficos salvos.")
    print(f"Tempo total: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    executar_experimento_completo_literatura()
