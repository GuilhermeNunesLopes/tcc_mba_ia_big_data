import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def process_log_anomalies(df_original, X_tfidf, y_true=None, model=None, contamination="auto"):
    """
    Identifica anomalias em logs numéricos aplicando um limiar dinâmico rigoroso.
    """
    df_result = df_original.copy()
    
    if len(df_result) != X_tfidf.shape[0]:
        raise ValueError(f"Dimensões incompatíveis: Logs ({len(df_result)}) vs TF-IDF ({X_tfidf.shape[0]})")

    if model is None:
        print("Treinando Isolation Forest com dados fornecidos...")
        # Ignoramos o parâmetro de contaminação nativo para usar nosso próprio limiar matemático
        model = IsolationForest(
            n_estimators=300,       
            max_samples=128, # Amostra menor força árvores mais rasas, ignorando micro-ruídos
            contamination='auto', 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_tfidf)
    
    # 1. Extração dos Scores Brutos (Quanto menor o score, mais anômalo)
    decision_scores = model.decision_function(X_tfidf)
    df_result['anomaly_score'] = decision_scores

    # ==========================================
    # CÁLCULO DO LIMIAR DINÂMICO DE SRE
    # ==========================================
    # Em vez de confiar no predict() cego, definimos que SOMENTE os 1.5% 
    # logs com os scores mais baixos absolutos do lote serão classificados como incidentes.
    # Você pode ajustar esse valor de 1.5 para 0.5 (mais rigoroso) ou 3.0 (mais sensível)
    percentil_corte = 1.5
    limiar_estatistico = np.percentile(decision_scores, percentil_corte)
    
    # Aplica a máscara: é anomalia apenas se o score for menor ou igual ao limite extremo
    df_result['pred_is_anomaly'] = (decision_scores <= limiar_estatistico).astype(int)
    # ==========================================
    
    if y_true is not None:
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
    print(f"Corte Dinâmico (Percentil {percentil_corte}%): Score <= {limiar_estatistico:.4f}")
    print("Contagem de Previsões:")
    print(df_result['pred_is_anomaly'].value_counts())
    print("\n" + "="*30)
    
    df_result = df_result.sort_values(by='anomaly_score', ascending=True)
    
    return df_result, model