import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def process_log_anomalies(df_original, X_tfidf, y_true=None, model=None, contamination="auto"):
    """
    Identifica anomalias em logs numéricos e calcula métricas se o ground truth for fornecido.
    """
    df_result = df_original.copy()
    
    if len(df_result) != X_tfidf.shape[0]:
        raise ValueError(f"Dimensões incompatíveis: Logs ({len(df_result)}) vs TF-IDF ({X_tfidf.shape[0]})")

    if model is None:
        print("Treinando Isolation Forest com dados fornecidos...")
        model = IsolationForest(
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

    df_result['pred_is_anomaly'] = (predictions == -1).astype(int)
    df_result['anomaly_score'] = decision_scores
    
    # Adicionamos o y_true ao DataFrame ANTES de calcular ou ordenar qualquer coisa
    if y_true is not None:
        # Garante compatibilidade caso y_true seja Series ou array
        df_result['y_true_label'] = y_true.values if isinstance(y_true, pd.Series) else y_true
        
        print("\n" + "="*30)
        print("Avaliação do Modelo vs Ground Truth:")
        
        metrix_confusion = confusion_matrix(df_result['y_true_label'], df_result['pred_is_anomaly'])
        precision = precision_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)
        recall = recall_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)
        f1 = f1_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)

        print("Matriz de Confusão:")
        print(metrix_confusion)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    print("\n" + "="*30)
    print("Contagem de Previsões:")
    print(df_result['pred_is_anomaly'].value_counts())
    print("\n" + "="*30)
    
    # Agora é seguro ordenar o DataFrame final pelo score
    df_result = df_result.sort_values(by='anomaly_score', ascending=True)
    
    return df_result, model