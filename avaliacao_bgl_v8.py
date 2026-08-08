import os
import sys
import json
import pandas as pd
import numpy as np
import tempfile
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Inserindo o diretório raiz no sys.path para garantir que 'modules' seja achado
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modules.preprocessor as preprocessor
import modules.parse_system as parse_system
import modules.anomaly_detector as anomaly_detector

CAMINHO_BGL = os.path.join("logpai", "BGL", "BGL.log")
SVD_COMPONENTS = [15, 30, 50]  

def carregar_bgl_rotulado(caminho: str = CAMINHO_BGL) -> pd.DataFrame:
    print("\n[0/4] Convertendo BGL bruto e extraindo templates via Drain3...")

    # 1. LEITURA OTIMIZADA COM PYTHON NATIVO E PANDAS
    # Usamos o leitor nativo do Python que é extremamente rápido e joga tudo no Pandas de uma vez
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        linhas = f.readlines()
        
    df_raw = pd.DataFrame({'linha_crua': linhas})
    
    # O Pandas faz o split vetorizado em C.
    # O .str.strip() remove os '\n' residuais antes do split.
    df = df_raw['linha_crua'].str.strip().str.split(r'\s+', n=9, expand=True)
    df.columns = ["Label", "ID", "Date_Alt", "Component_1", "Timestamp", "Component_2", "Subsystem", "Level", "Type", "Content"]
    
    # Libera memória apagando os dados brutos
    del linhas
    del df_raw 
    df = df.dropna(subset=['Content']).reset_index(drop=True)

    # 2. ESCRITA OTIMIZADA PARA O DRAIN3
    fd, temp_path = tempfile.mkstemp(text=True, suffix=".log")
    with open(fd, 'w', encoding='utf-8') as f:
        # Join nativo em memória (muito rápido)
        f.write('\n'.join(df["Content"].astype(str).tolist()) + '\n')

    print("Processando arquivo temporário com o Drain3...")
    df_parsed = parse_system.automatic_drain_parse(
        file_path=temp_path,
        nome_fonte="BGL_Eval"
    )

    os.remove(temp_path)

    if len(df_parsed) != len(df):
        raise ValueError(f"Divergência no parser: Original={len(df)} linhas vs Parsed={len(df_parsed)} linhas.")

    # 3. CONVERSÃO DE DATAS
    df_timestamp = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d-%H.%M.%S.%f", errors='coerce')
    
    if df_timestamp.isna().all():
        df_timestamp = pd.to_datetime(df["Timestamp"], errors='coerce')

    # 4. CONSTRUÇÃO DO DATAFRAME FINAL
    df_padronizado = pd.DataFrame({
        "Timestamp": df_timestamp,
        "Level": df["Level"].astype(str),
        "Source": df["Component_1"].astype(str),
        "Event": df_parsed["Template"].astype(str), 
        "Raw_Log": df["Level"].astype(str) + " " + df_parsed["Template"].astype(str),
        "Label_Original": df["Label"].astype(str),
    })

    df_padronizado["y_true"] = (df_padronizado["Label_Original"] != "-").astype(int)
    df_padronizado = df_padronizado.dropna(subset=["Timestamp"])
    df_padronizado = df_padronizado.sort_values("Timestamp").reset_index(drop=True)

    return df_padronizado

def main():
    print("=" * 62)
    print("AVALIAÇÃO CIENTÍFICA — Pipeline de Detecção de Anomalias")
    print("Abordagem: Time Series Split + TF-IDF + SVD + iForest vs SVM")
    print("=" * 62)

    df_bruto = carregar_bgl_rotulado()

    print("\nCalculando features temporais (Log Burst e Time Diff)...")
    df_bruto['time_diff'] = df_bruto['Timestamp'].diff().dt.total_seconds().fillna(0)
    df_bruto_idx = df_bruto.set_index('Timestamp')
    df_bruto['rolling_count'] = df_bruto_idx.index.to_series().rolling('60s').count().values
    
    # Validação cruzada temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    resultados_finais = {}

    for n_comp in SVD_COMPONENTS:
        print(f"\n==================================================")
        print(f"Executando Experimento com SVD Componentes = {n_comp}")
        print(f"==================================================")
        
        fold = 1
        historico_f1_iforest = []
        historico_f1_svm = []
        
        for train_index, test_index in tscv.split(df_bruto):
            print(f"\n--- Processando Fold {fold} ---")
            df_train = df_bruto.iloc[train_index].reset_index(drop=True)
            df_test = df_bruto.iloc[test_index].reset_index(drop=True)
            
            y_train = df_train['y_true'].values
            y_test = df_test['y_true'].values
            
            # TF-IDF
            X_train_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_train)
            X_test_tfidf, _ = preprocessor.tfidf_vectorize(df_test, vectorizer=vectorizer)
            
            # SVD
            X_train_svd, svd_model = preprocessor.apply_truncated_svd(X_train_tfidf, n_components=n_comp)
            X_test_svd, _ = preprocessor.apply_truncated_svd(X_test_tfidf, svd_model=svd_model, n_components=n_comp)
            
            # Normalização Temporal
            scaler = StandardScaler()
            features_tempo_treino = scaler.fit_transform(df_train[['time_diff', 'rolling_count']].values)
            features_tempo_teste = scaler.transform(df_test[['time_diff', 'rolling_count']].values)
            
            X_train_final = np.hstack((X_train_svd, features_tempo_treino))
            X_test_final = np.hstack((X_test_svd, features_tempo_teste))
            
            print(">>> Otimizando e Treinando Modelos...")
            # iForest
            modelo_iforest, _, thresh_iforest = anomaly_detector.optimize_isolation_forest(X_train_final, y_train)
            scores_iforest = modelo_iforest.decision_function(X_test_final)
            pred_iforest = np.where(-scores_iforest >= thresh_iforest, 1, 0)
            
            ## SVM
            modelo_svm, _, thresh_svm = anomaly_detector.optimize_one_class_svm(X_train_final, y_train)
            scores_svm = modelo_svm.decision_function(X_test_final)
            pred_svm = np.where(-scores_svm >= thresh_svm, 1, 0)
            #
            # F1-Scores
            f1_iforest = f1_score(y_test, pred_iforest, zero_division=0)
            f1_svm = f1_score(y_test, pred_svm, zero_division=0)
            #
            historico_f1_iforest.append(f1_iforest)
            historico_f1_svm.append(f1_svm)
            
            print(f"F1-Score iForest: {f1_iforest:.4f} | F1-Score SVM: {f1_svm:.4f}")
            fold += 1
            
        resultados_finais[f"SVD_{n_comp}"] = {
            "iForest_F1_Medio": float(np.mean(historico_f1_iforest)),
            "SVM_F1_Medio": float(np.mean(historico_f1_svm))
        }

    print("\n" + "="*62)
    print("MÉTRICAS FINAIS CONSOLIDADAS DO PIPELINE (Time Series Split)")
    print("=" * 62)
    print(json.dumps(resultados_finais, indent=4))

if __name__ == "__main__":
    main()