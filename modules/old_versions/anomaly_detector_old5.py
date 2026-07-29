import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc


def process_log_anomalies(df_original, X_tfidf, y_true=None, model=None, contamination="auto", anomaly_percentile=1.5):
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
            max_samples='auto',  # Usar todos os dados para treinar cada árvore
            #max_samples=128, # Amostra menor força árvores mais rasas, ignorando micro-ruídos
            contamination='auto', 
            random_state=43, 
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
    limiar_estatistico = np.percentile( decision_scores, anomaly_percentile)
    
    # Aplica a máscara: é anomalia apenas se o score for menor ou igual ao limite extremo
    df_result['pred_is_anomaly'] = (decision_scores <= limiar_estatistico).astype(int)
    # ==========================================
    
    metricas_calculadas = {}

    if y_true is not None:
        df_result['y_true_label'] = y_true.values if isinstance(y_true, pd.Series) else y_true
        scores_para_curva = -df_result['anomaly_score'].values
        precisions, recalls, thresholds = precision_recall_curve(df_result['y_true_label'], scores_para_curva)
        pr_auc = auc(recalls, precisions)

        print("\n" + "="*30)
        print("Avaliação do Modelo vs Ground Truth:")
        
        metrix_confusion = confusion_matrix(df_result['y_true_label'], df_result['pred_is_anomaly'])
        precision = precision_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)
        recall = recall_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)
        f1 = f1_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)

        # Salva em um dicionário para enviar ao Streamlit
        metricas_calculadas = {
            "PR_AUC": float(pr_auc),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1_Score": float(f1)
        }

    df_result = df_result.sort_values(by='anomaly_score', ascending=True)
    
    # Adicionamos a variável 'metricas_calculadas' no return (agora são 3 itens)
    return df_result, model, metricas_calculadas