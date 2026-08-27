import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def process_log_anomalies(df_original, X_tfidf, y_true=None, model=None, contamination="auto"):
    """
    Identifica anomalias em logs numéricos e calcula métricas se o ground truth for fornecido.
    
    y_true: Série ou array com os rótulos REAIS das anomalias (1 para anomalia, 0 para normal).
    """
    df_result = df_original.copy()
    
    if len(df_result) != X_tfidf.shape[0]:
        raise ValueError(f"Dimensões incompatíveis: Logs ({len(df_result)}) vs TF-IDF ({X_tfidf.shape[0]})")

    if model is None:
        print("Treinando Isolation Forest com dados fornecidos...")
        model = IsolationForest(
            # test de parâmetros: https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html
            n_estimators=300,       
            max_samples='auto',
            contamination=contamination, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_tfidf)
    
    # Predição e Extração de Scores
    predictions = model.predict(X_tfidf)
    decision_scores = model.decision_function(X_tfidf)

    # Scikit-Learn: 1 = Normal, -1 = Anomalia. Vamos converter para 0 (Normal) e 1 (Anomalia) 
    # para ficar no padrão clássico de métricas binárias.
    df_result['pred_is_anomaly'] = (predictions == -1).astype(int)
    df_result['anomaly_score'] = decision_scores
    
    df_result = df_result.sort_values(by='anomaly_score', ascending=True)
    
    # Só calcula métricas se você tiver o Ground Truth para comparar
    if y_true is not None:
        print("\n" + "="*30)
        print("Avaliação do Modelo vs Ground Truth:")
        
        # Garante que y_true está no formato correto (1 para anomalia, 0 normal)
        metrix_confusion = confusion_matrix(y_true, df_result['pred_is_anomaly'])
        precision = precision_score(y_true, df_result['pred_is_anomaly'], zero_division=0)
        recall = recall_score(y_true, df_result['pred_is_anomaly'], zero_division=0)
        f1 = f1_score(y_true, df_result['pred_is_anomaly'], zero_division=0)

        print("Matriz de Confusão:")
        print(metrix_confusion)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    print("\n" + "="*30)
    print("Contagem de Previsões:")
    print(df_result['pred_is_anomaly'].value_counts())
    print("\n" + "="*30)
    
    return df_result, model