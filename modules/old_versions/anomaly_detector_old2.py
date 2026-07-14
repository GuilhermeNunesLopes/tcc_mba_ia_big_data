import pandas as pd
import numpy as np
import scipy.sparse as sp # Importação opcional, boa para verificações futuras
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def process_log_anomalies(df_original, X_tfidf, model=None, contamination="auto"):
    """
    Identifica anomalias em logs numéricos.
    Agora otimizado para receber X_tfidf como uma MATRIZ ESPARSA (SciPy) 
    direto do preprocessor, ao invés de um DataFrame pesado do Pandas.
    """
    df_result = df_original.copy()
    
    # Check de consistência: O atributo .shape funciona perfeitamente em matrizes esparsas
    if len(df_result) != X_tfidf.shape[0]:
        raise ValueError(f"Dimensões incompatíveis: Logs ({len(df_result)}) vs TF-IDF ({X_tfidf.shape[0]})")

    # Se não existe modelo pré-treinado, instanciamos e treinamos um
    if model is None:
        # OTIMIZAÇÃO: n_estimators reduzido para 50 para decisões ultra-rápidas
        print("Treinando Isolation Forest com dados de treino...")
        model = IsolationForest(
            n_estimators=200,  
            #n_estimators=50,       
            max_samples='auto',
            contamination=contamination, 
            random_state=42, 
            n_jobs=-1
        )
        # O Scikit-Learn processa a matriz esparsa nativamente sem estourar a memória
        model.fit(X_tfidf)
    
    # Predição e Extração de Scores (Ultra-rápido com matriz esparsa)
    predictions = model.predict(X_tfidf)
    decision_scores = model.decision_function(X_tfidf)

   
    # Injeção de resultados no DataFrame
    df_result['anomaly_label'] = predictions
    
    # FORMA CORRETA E RÁPIDA NO PANDAS (Vetorizada)
    df_result['is_anomaly'] = df_result['anomaly_label'] == -1
    
    df_result['anomaly_score'] = decision_scores
    
    df_result = df_result.sort_values(by='anomaly_score', ascending=True)
    # Retorna o dataframe enriquecido e também o modelo salvo!
    
    print("\n" + "="*30)
    print("Resumo de Anomalias Detectadas:")
    metrix_confusion = confusion_matrix(df_result['is_anomaly'], df_result['anomaly_label'] == -1)
    precision = precision_score(df_result['is_anomaly'], df_result['anomaly_label'] == -1)
    recall = recall_score(df_result['is_anomaly'], df_result['anomaly_label'] == -1)
    f1 = f1_score(df_result['is_anomaly'], df_result['anomaly_label'] == -1)

    print("\n" + "="*30)
    print("Resumo de Anomalias Detectadas:")
    print("Matriz de Confusão:")
    print(metrix_confusion)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    print("\n" + "="*30)
    print(df_result['is_anomaly'].value_counts())
    print("\n" + "="*30)
    
    return df_result, model


