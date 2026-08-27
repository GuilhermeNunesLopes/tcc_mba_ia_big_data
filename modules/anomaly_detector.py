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

    param_grid = {
        "n_estimators": [300, 500],
        "max_samples": [256, 512, "auto"],
        "max_features": [1.0],
        "bootstrap": [False, True]
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

        beta = 1.0
        fbeta_scores = ((1 + beta**2) * precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-10)

        # PERFORMANCE: a versão original recalculava (scores_v >= t).mean()
        # varrendo o array de validação inteiro para CADA threshold — O(n*m),
        # com m também crescendo com n (precision_recall_curve gera ~1
        # threshold por score único). Em 157 mil linhas de validação isso já
        # mede ~19s por combinação do grid (confirmado empiricamente); em
        # folds de centenas de milhares a milhões de linhas (como o BGL
        # completo) o mesmo cálculo passa de horas para dias, tornando o
        # grid search inviável. A versão vetorizada abaixo ordena os scores
        # uma vez (O(n log n)) e usa busca binária (searchsorted) para achar,
        # para todos os thresholds de uma vez, quantos scores são >= threshold
        # — resultado numericamente idêntico (validado célula a célula),
        # ~2400x mais rápido no teste com 157 mil linhas.
        scores_v_ordenados = np.sort(scores_v)
        posicoes = np.searchsorted(scores_v_ordenados, thresholds, side='left')
        taxas_anomalia = (len(scores_v) - posicoes) / len(scores_v)
        fbeta_scores[taxas_anomalia > 0.10] = 0.0

        idx = np.argmax(fbeta_scores)
        current_threshold = thresholds[idx]
        pred_v = (scores_v >= current_threshold).astype(int)

        f1 = f1_score(y_v, pred_v)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_threshold = current_threshold
            # Mantém o modelo já treinado em X_t — é ele que gerou os escores
            # usados para escolher best_threshold acima. Retreinar em X_train
            # completo aqui criaria um modelo novo com uma distribuição de
            # escore diferente, desalinhado do threshold escolhido.
            best_model = model

    print("\n===== Melhor configuração =====")
    print(best_params)
    print(f"Threshold: {best_threshold:.4f}")
    print(f"F1 (Validação): {best_f1:.4f}")

    return best_model, best_params, best_threshold

def process_log_anomalies(df_original, X_tfidf, y_true=None, model=None, best_threshold=None, contamination="auto", anomaly_percentile=3, algorithm="iforest"):
    
    df_result = df_original.copy()
    
    if len(df_result) != X_tfidf.shape[0]:
        raise ValueError(f"Dimensões incompatíveis: Logs ({len(df_result)}) vs TF-IDF ({X_tfidf.shape[0]})")

    # Guarda a posição posicional original (0-based) ANTES de qualquer reordenação.
    # É isso que permite, depois do sort por anomaly_score no final desta função,
    # mapear cada linha de volta para a linha correspondente em X_tfidf/tfidf_test
    # (por exemplo, para explicabilidade por termos ou para o DBSCAN em main_v4.py).
    df_result['_row_pos'] = np.arange(len(df_result))
    
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

        # Mesma otimização O(n log n) aplicada em optimize_isolation_forest —
        # ver o comentário lá para a explicação completa do problema O(n*m).
        scores_v_ordenados = np.sort(scores_v)
        posicoes = np.searchsorted(scores_v_ordenados, thresholds, side='left')
        taxas_anomalia = (len(scores_v) - posicoes) / len(scores_v)
        fbeta_scores[taxas_anomalia > 0.10] = 0.0

        idx = np.argmax(fbeta_scores)
        current_threshold = thresholds[idx]
        pred_v = (scores_v >= current_threshold).astype(int)

        f1 = f1_score(y_v, pred_v, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_threshold = current_threshold
            # Mesmo raciocínio do Isolation Forest: mantém o modelo já
            # treinado em X_t, que é o que gerou os escores do threshold.
            best_model = model

    print("\n===== Melhor configuração OCSVM =====")
    print(best_params)
    print(f"Threshold: {best_threshold:.4f}")
    print(f"F1 (Validação): {best_f1:.4f}")

    return best_model, best_params, best_threshold