import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
#from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
    auc
)
from sklearn.svm import OneClassSVM

def optimize_isolation_forest(X_train, y_train):
    """
    Otimiza o Isolation Forest evitando vazamento de dados.
    Divide os dados em treino interno e validação.
    """
    # SPLIT INTERNO: Evita avaliar no mesmo dado em que treinou
    X_t, X_v, y_t, y_v = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )

    #param_grid = {
    #     "n_estimators": [300], #melhor config
    #    #"n_estimators": [100, 200, 300, 500],
    #    #"max_samples": [128, 256, 512, "auto"],
    #    "max_samples": [128], #melhor config
    #    "max_features": [0.6],#melhor config
    #    #"max_features": [0.6, 0.8, 1.0],
    #    #"bootstrap": [True, False],
    #    "bootstrap": [True], #melhor config
    #}
    param_grid = {
    "n_estimators": [300, 500],
    "max_samples": [256, 512,"auto"],
    #"max_features": [0.2, 0.5, 1.0], 
    "max_features": [1.0],    
    "bootstrap": [False]
    }

    best_model = None
    best_params = None
    best_threshold = None
    best_f1 = -1

    for params in ParameterGrid(param_grid):
        model = IsolationForest(
            contamination="auto",
            random_state=42,
            n_jobs=-1,
            **params,
        )

        # Treina APENAS na fatia interna de treino
        model.fit(X_t)

        # Avalia APENAS na fatia de validação
        scores_v = -model.decision_function(X_v)

        precisions, recalls, thresholds = precision_recall_curve(y_v, scores_v)

        # Calcula F1 ignorando divisões por zero de forma segura
        beta = 1.0
        #beta = 0.5  # Dá mais peso à precisão, visto que falsos positivos são mais críticos
        fbeta_scores = ((1 + beta**2) * precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-10)

        taxas_anomalia = np.array([(scores_v >= t).mean() for t in thresholds])
        fbeta_scores[taxas_anomalia > 0.10] = 0.0

        idx = np.argmax(fbeta_scores)
        current_threshold = thresholds[idx]
        pred_v = (scores_v >= current_threshold).astype(int)
        
        f1 = f1_score(y_v, pred_v)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_threshold = current_threshold
            
            # Retreina o melhor modelo com TODO o dado de treino original (X_train completo)
            best_model = IsolationForest(contamination="auto", random_state=42, n_jobs=-1, **params)
            best_model.fit(X_train)

    print("\n===== Melhor configuração =====")
    print(best_params)
    print(f"Threshold: {best_threshold:.4f}")
    print(f"F1 (Validação): {best_f1:.4f}")

    return best_model, best_params, best_threshold


def process_log_anomalies(df_original, X_tfidf, y_true=None, model=None, best_threshold=None, contamination="auto", anomaly_percentile=3, algorithm="iforest"):
    
    df_result = df_original.copy()
    
    if len(df_result) != X_tfidf.shape[0]:
        raise ValueError(f"Dimensões incompatíveis: Logs ({len(df_result)}) vs TF-IDF ({X_tfidf.shape[0]})")
    
    best_params_out = None
    
    # FASE DE TREINO
    if model is None:
        print(f"Fase de Treino: Otimizando {algorithm.upper()}...")
        if y_true is not None:
            if algorithm == "iforest":
                model, best_params_out, best_threshold = optimize_isolation_forest(X_tfidf, y_true)
            elif algorithm == "ocsvm":
                model, best_params_out, best_threshold = optimize_one_class_svm(X_tfidf, y_true)
            else:
                raise ValueError("Algoritmo desconhecido. Escolha 'iforest' ou 'ocsvm'.")
        else:
            # Fallback genérico caso rode em produção sem label
            #model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42) if algorithm == "iforest" else OneClassSVM(nu=0.05)
            model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42) if algorithm == "iforest" else OneClassSVM(kernel='linear', nu=contamination)
            model.fit(X_tfidf)
    else:
        print(f"Fase de Teste: Usando modelo {algorithm.upper()} e threshold previamente treinados.")

    # ... O restante da função continua exatamente igual a partir de "1. Extração dos Scores Brutos"[cite: 10]
    # 1. Extração dos Scores Brutos
    decision_scores = model.decision_function(X_tfidf)
    df_result['anomaly_score'] = decision_scores

    # CÁLCULO / APLICAÇÃO DO LIMIAR
    if best_threshold is not None:
        df_result["pred_is_anomaly"] = (-decision_scores >= best_threshold).astype(int)
    else:
        limiar_estatistico = np.percentile(decision_scores, anomaly_percentile)
        df_result["pred_is_anomaly"] = (decision_scores <= limiar_estatistico).astype(int)

    metricas_calculadas = {}

    if y_true is not None:
        df_result['y_true_label'] = y_true.values if isinstance(y_true, pd.Series) else y_true
        scores_para_curva = -df_result['anomaly_score'].values
        precisions, recalls, thresholds_pr = precision_recall_curve(df_result['y_true_label'], scores_para_curva)
        pr_auc = auc(recalls, precisions)

        print("\n" + "="*30)
        print("Avaliação do Modelo vs Ground Truth:")
        
        precision = precision_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)
        recall = recall_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)
        f1 = f1_score(df_result['y_true_label'], df_result['pred_is_anomaly'], zero_division=0)

        metricas_calculadas = {
            "PR_AUC": float(pr_auc),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1_Score": float(f1)
        }

    df_result = df_result.sort_values(by='anomaly_score', ascending=True)
    
    return df_result, model, metricas_calculadas, best_params_out, best_threshold



############################################
#   TESTANDO OUTRO MODELO DE DETECÇÃO DE ANOMALIAS (ONE-CLASS SVM)
############################################
def optimize_one_class_svm(X_train, y_train):
    """
    Otimiza o One-Class SVM com a mesma métrica F-Beta do Isolation Forest.
    """
    X_t, X_v, y_t, y_v = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )

    # Hiperparâmetros matemáticos da fronteira do SVM
    param_grid = {
        "kernel": ["linear", "rbf"], # O Linear costuma destruir o RBF em matrizes esparsas de texto
        "gamma": ["scale", "auto"],  # Usado apenas se o RBF for escolhido
        "nu": [0.05, 0.08, 0.12]     # nu deve abraçar a taxa de anomalias real do dataset de treino (7.7%)
    }

    best_model = None
    best_params = None
    best_threshold = None
    best_f1 = -1

    for params in ParameterGrid(param_grid):
        model = OneClassSVM(**params)

        # Treina APENAS na fatia interna
        model.fit(X_t)

        # O decision_function do SVM funciona igual ao do IF: valores negativos são anomalias
        # Multiplicamos por -1 para a curva PR_AUC funcionar corretamente
        scores_v = -model.decision_function(X_v)

        precisions, recalls, thresholds = precision_recall_curve(y_v, scores_v)

        # Usando Beta = 1.0 (F1-Score puro) para ser a mesma base de comparação do IF
        beta = 1.0
        fbeta_scores = ((1 + beta**2) * precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-10)

        taxas_anomalia = np.array([(scores_v >= t).mean() for t in thresholds])
        fbeta_scores[taxas_anomalia > 0.10] = 0.0

        idx = np.argmax(fbeta_scores)
        current_threshold = thresholds[idx]
        pred_v = (scores_v >= current_threshold).astype(int)
        
        f1 = f1_score(y_v, pred_v, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_threshold = current_threshold
            
            # Retreina o melhor modelo com TODO o dado de treino
            best_model = OneClassSVM(**params)
            best_model.fit(X_train)

    print("\n===== Melhor configuração OCSVM =====")
    print(best_params)
    print(f"Threshold: {best_threshold:.4f}")
    print(f"F1 (Validação): {best_f1:.4f}")

    return best_model, best_params, best_threshold
