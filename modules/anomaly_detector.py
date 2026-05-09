import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def process_log_anomalies(df_original, X_tfidf, contamination=0.02):
    """
    Recebe os logs parseados e a matriz numérica para identificar anomalias.
    Retorna o DataFrame original enriquecido com os scores do modelo.
    """
    # 1. Trabalhar em uma cópia para preservar os dados originais
    df_result = df_original.copy()
    
    # 2. Check de consistência: essencial para evitar erros de índice no TCC
    if len(df_result) != X_tfidf.shape[0]:
        raise ValueError(f"Dimensões incompatíveis: Logs ({len(df_result)}) vs TF-IDF ({X_tfidf.shape[0]})")

    # 3. Configuração do Modelo
    # n_jobs=-1 utiliza todo o processamento disponível da sua máquina
    model = IsolationForest(
        contamination=contamination, 
        random_state=42, 
        n_jobs=-1
    )
    
    # 4. Treinamento e Predição
    # O modelo olha apenas para a matriz TF-IDF (X_tfidf)
    predictions = model.fit_predict(X_tfidf)
    
    # 5. Extração de Scores de Decisão
    # Quanto menor (mais negativo) o score, mais anômalo é o log
    decision_scores = model.decision_function(X_tfidf)
    
    # 6. Injeção de resultados no DataFrame
    df_result['anomaly_label'] = predictions # 1 para normal, -1 para anomalia
    df_result['is_anomaly'] = df_result['anomaly_label'].apply(lambda x: True if x == -1 else False)
    df_result['anomaly_score'] = decision_scores
    
    return df_result