import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import sys

# 1. Descobre o caminho absoluto da pasta atual onde o script está
pasta_atual = os.path.dirname(os.path.abspath(__file__))

# 2. Volta um nível de diretório (pasta pai)
pasta_pai = os.path.abspath(os.path.join(pasta_atual, '..'))

# 3. Adiciona a pasta pai no topo da lista de caminhos do Python
if pasta_pai not in sys.path:
    sys.path.insert(0, pasta_pai)

# Importando da sua arquitetura modular
from modules.anomaly_detector import process_log_anomalies
from modules.metrics import calculate_metrics

def executar_pipeline_anomalias():
    print("="*60)
    print("🚀 INICIANDO TREINO E DETECÇÃO DE ANOMALIAS (HDFS 50/50)")
    print("="*60)

    path_treino = 'resultados/hdfs_treino_normal.parquet'
    path_teste = 'resultados/hdfs_teste_50_50.parquet'
    
    if not os.path.exists(path_treino) or not os.path.exists(path_teste):
        print("⚠️ ERRO: Execute primeiro o script gerador_experimento_hdfs.py")
        return

    # 1. CARREGAMENTO DOS DADOS
    df_treino = pd.read_parquet(path_treino)
    df_teste = pd.read_parquet(path_teste)

    # Separar as features (contagem de eventos) das colunas de controle
    X_treino_raw = df_treino.drop(columns=['BlockId', 'Label'], errors='ignore')
    X_teste_raw = df_teste.drop(columns=['BlockId', 'Label'], errors='ignore')

    # Garantir que o dataset de teste tenha exatamente as mesmas features do treino
    X_teste_raw = X_teste_raw.reindex(columns=X_treino_raw.columns, fill_value=0)

    # 2. PRÉ-PROCESSAMENTO (TF-IDF)
    print("\n[1/3] Aplicando vetorização TF-IDF na matriz de frequência...")
    # Como já temos uma matriz de contagem pronta, usamos o Transformer do scikit-learn
    # em vez do preprocessor.py, que é otimizado para lidar com strings de texto livre.
    tfidf = TfidfTransformer()
    X_treino_tfidf = tfidf.fit_transform(X_treino_raw)
    X_teste_tfidf = tfidf.transform(X_teste_raw)

    # 3. TREINAMENTO (APENAS COM LOGS NORMAIS)
    print("\n[2/3] Fase de Treinamento (Isolation Forest)...")
    # Passamos y_true=None para acionar o fit genérico não supervisionado do anomaly_detector
    _, modelo_treinado, _, _, _ = process_log_anomalies(
        df_original=df_treino, 
        X_tfidf=X_treino_tfidf, 
        y_true=None, 
        algorithm="iforest",
        contamination=0.01 
    )

    # 4. TESTE E DETECÇÃO (NO DATASET 50/50)
    print("\n[3/3] Fase de Teste (Validação no Dataset Balanceado)...")
    
    # Injetamos o modelo treinado. Como não otimizamos um threshold de F1 no treino, 
    # o módulo usará o np.percentile para cortar as anomalias baseado no percentil.
    # Ajustamos anomaly_percentile=50 pois sabemos que o dataset é 50/50.
    df_resultado, _, _, _, _ = process_log_anomalies(
        df_original=df_teste, 
        X_tfidf=X_teste_tfidf, 
        y_true=df_teste['Label'], 
        model=modelo_treinado, 
        algorithm="iforest",
        anomaly_percentile=50 
    )

    # 5. AVALIAÇÃO DE DESEMPENHO
    print("\n" + "="*40)
    print("📊 RESULTADOS DO EXPERIMENTO")
    print("="*40)
    
    # Extrai as métricas usando o módulo customizado metrics.py
    precision, recall, f1, cm = calculate_metrics(
        y_true=df_resultado['y_true_label'], 
        y_pred=df_resultado['pred_is_anomaly']
    )
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nMatriz de Confusão:")
    print(cm)
    
    df_resultado.to_parquet('resultados/resultado_final_hdfs.parquet', index=False)
    print("\n✅ Predições e Scores salvos em 'resultados/resultado_final_hdfs.parquet'")

if __name__ == "__main__":
    executar_pipeline_anomalias()
