import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def process_log_anomalies(df_original, X_tfidf, model=None, contamination=0.02):
    """
    Identifica anomalias em logs numéricos.
    Se 'model' não for passado, cria e treina um novo (Modo Treino).
    Se 'model' for passado, apenas aplica o modelo existente (Modo Inferência).
    """
    df_result = df_original.copy()
    
    # Check de consistência
    if len(df_result) != X_tfidf.shape[0]:
        raise ValueError(f"Dimensões incompatíveis: Logs ({len(df_result)}) vs TF-IDF ({X_tfidf.shape[0]})")

    # Se não existe modelo pré-treinado, instanciamos e treinamos um
    if model is None:
        model = IsolationForest(
            contamination=contamination, 
            random_state=42, 
            n_jobs=-1
        )
        # Ajusta o modelo (aprende o que é "normal")
        model.fit(X_tfidf)
    
    # Predição e Extração de Scores (Funciona tanto para treino quanto para dados novos)
    predictions = model.predict(X_tfidf)
    decision_scores = model.decision_function(X_tfidf)
    
    # Injeção de resultados no DataFrame
    df_result['anomaly_label'] = predictions
    
    # FORMA CORRETA E RÁPIDA NO PANDAS (Vetorizada)
    df_result['is_anomaly'] = df_result['anomaly_label'] == -1
    
    df_result['anomaly_score'] = decision_scores
    
    # Retorna o dataframe enriquecido e também o modelo salvo!
    return df_result, model


def calculate_metrics(y_true, y_pred):
    """
    Calcula as métricas científicas do modelo.
    y_true: Série com os labels REAIS (True para anomalia, False para normal)
    y_pred: Série com os labels PREDITOS pelo Isolation Forest
    """
    # zero_division=0 evita erros se o modelo não detectar nenhuma anomalia
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return precision, recall, f1, cm