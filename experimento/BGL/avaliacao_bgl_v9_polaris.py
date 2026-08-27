import os
import sys
import json
import argparse
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
import polars as pl

# 1. Descobre o caminho absoluto da pasta atual onde o script está
pasta_atual = os.path.dirname(os.path.abspath(__file__))

# 2. Volta um nível de diretório (pasta pai)
pasta_pai = os.path.abspath(os.path.join(pasta_atual, '../../'))

# 3. Adiciona a pasta pai no topo da lista de caminhos do Python
if pasta_pai not in sys.path:
    sys.path.insert(0, pasta_pai)

import modules.preprocessor as preprocessor
import modules.parse_system as parse_system
import modules.anomaly_detector as anomaly_detector
import modules.run_output as run_output

CAMINHO_BGL = os.path.join("logpai", "BGL", "BGL.log")
SVD_COMPONENTS = [15, 30, 50]

# BGL.log é o dataset bruto completo (~4,7 milhões de linhas) — nos folds
# finais do TimeSeriesSplit o treino cresce até abranger quase esse total.
# O OneClassSVM (libsvm) tem custo mais que quadrático no número de amostras
# (mesmo problema já documentado/corrigido em experimento_pipeline_svm.py,
# achado 3.5 da Validação de Resultados v2) — sem limite, o treino do OCSVM
# em folds grandes não terminaria em tempo hábil. O Isolation Forest não tem
# essa limitação e continua treinando no fold inteiro.
LIMITE_AMOSTRA_OCSVM = 20000

def carregar_bgl_rotulado(caminho: str = CAMINHO_BGL, limite_linhas: int = None, resumo_saida=None) -> pd.DataFrame:
    print("\n[0/4] Convertendo BGL bruto e extraindo templates via Drain3...")
    if limite_linhas:
        print(f"    (limite de teste: lendo só as primeiras {limite_linhas} linhas — ver --limite-linhas)")

    # 1. LEITURA ULTRA-RÁPIDA COM POLARS
    # Lê as linhas do arquivo mantendo a tolerância a erros de codificação
    # n_rows corta a cauda do arquivo mantendo a ordem cronológica original
    # (o TimeSeriesSplit exige ordem) — None lê o arquivo inteiro.
    df_pl = pl.read_csv(
            caminho,
            has_header=False,
            separator="\x1e",
            new_columns=["column_1"],
            encoding="utf8-lossy",
            truncate_ragged_lines=True,
            n_rows=limite_linhas,
        )
    # Expressão regular para replicar o r'\s+' com n=9. 
    regex_bgl = r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$"
    
    # Aplica a extração via engine regex multithreaded do Polars e desaninha a estrutura
    df_pl = df_pl.select(
        pl.col("column_1").str.extract_groups(regex_bgl)
    ).unnest("column_1")
    
    # Renomeia as colunas
    df_pl.columns = ["Label", "ID", "Date_Alt", "Component_1", "Timestamp", "Component_2", "Subsystem", "Level", "Type", "Content"]

    # Converte de volta para o Pandas (operação zero-copy sempre que possível) 
    # para garantir compatibilidade com o restante do pipeline original
    df = df_pl.to_pandas()
    df = df.dropna(subset=['Content']).reset_index(drop=True)

    # 2. ESCRITA OTIMIZADA PARA O DRAIN3
    fd, temp_path = tempfile.mkstemp(text=True, suffix=".log")
    with open(fd, 'w', encoding='utf-8') as f:
        # Join nativo em memória
        f.write('\n'.join(df["Content"].astype(str).tolist()) + '\n')

    print("Processando arquivo temporário com o Drain3...")
    
    # -------------------------------------------------------------
    # LEMBRETE: Aqui aplicamos a correção anterior do gerador!
    print("Processando arquivo temporário com o Drain3...")

    # ACHADO CORRIGIDO: nome de fonte isolado do motor ao vivo (evita
    # compartilhar/"contaminar" o estado do Drain3 usado pela ingestão real)
    # e reset do estado de uma execução anterior desta MESMA avaliação antes
    # de começar — mesmo padrão já aplicado em avaliacao_walkforward.py.
    # Nome próprio (diferente do v8) para que rodar v8 e v9 na mesma sessão
    # não faça um contaminar o estado do outro.
    nome_fonte_isolado = "BGL_Eval_v9_isolado"
    caminho_estado = os.path.join(
        parse_system.DIRETORIO_ESTADOS_PADRAO, f"drain3_state_{nome_fonte_isolado}.bin"
    )
    if os.path.isfile(caminho_estado):
        os.remove(caminho_estado)

    # 3. EXTRAÇÃO ROBUSTA EM LOTES (BATCHES) DO DRAIN3
    gerador_parsed = parse_system.automatic_drain_parse(
        file_path=temp_path,
        nome_fonte=nome_fonte_isolado,
        resumo_saida=resumo_saida,
    )
    
    templates_limpos = []
    
    # O gerador yielda blocos (lotes) de dados, precisamos "desempacotar" cada um
    for lote in gerador_parsed:
        
        # Prevenção caso o lote venha como DataFrame em vez de lista
        if isinstance(lote, pd.DataFrame):
            itens = lote.to_dict('records')
        else:
            itens = lote
            
        # Itera linha a linha dentro do lote
        for item in itens:
            if isinstance(item, dict):
                templates_limpos.append(item.get("Template", item.get("template_mined", "")))
            elif isinstance(item, (list, tuple)):
                try:
                    templates_limpos.append(str(item[0]))
                except IndexError:
                    templates_limpos.append("")
            else:
                templates_limpos.append(str(item))

    # Cria o DataFrame consolidado com todas as 4.7 milhões de linhas
    df_parsed = pd.DataFrame({"Template": templates_limpos})
    # -------------------------------------------------------------

    os.remove(temp_path)

    if len(df_parsed) != len(df):
        raise ValueError(f"Divergência no parser: Original={len(df)} linhas vs Parsed={len(df_parsed)} linhas.")

    print(f"   -> Linhas brutas lidas do BGL.log: {len(df)}")
    print(f"   -> Linhas após parse via Drain3:   {len(df_parsed)}")

    # 3. CONVERSÃO DE DATAS (Mantém a lógica do Pandas original)
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

    print(f"   -> Linhas finais após limpeza/conversão de datas: {len(df_padronizado)}")

    return df_padronizado
def main(limite_linhas=None):
    print("=" * 62)
    print("AVALIAÇÃO CIENTÍFICA — Pipeline de Detecção de Anomalias")
    print("Abordagem: Time Series Split + TF-IDF + SVD + iForest vs SVM")
    print("=" * 62)

    # resumo_parse é preenchido em memória por carregar_bgl_rotulado() (via
    # automatic_drain_parse, ver modules/parse_system.py) com o resumo
    # antes/depois do parse Drain3 (linhas brutas -> templates únicos) — para
    # salvar junto do resultado desta execução, não só imprimir no console.
    resumo_parse = {}
    df_bruto = carregar_bgl_rotulado(limite_linhas=limite_linhas, resumo_saida=resumo_parse)

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
            
            ## SVM — amostra o treino se o fold exceder LIMITE_AMOSTRA_OCSVM
            # (nos folds finais do TimeSeriesSplit sobre o BGL.log completo,
            # o treino pode chegar a milhões de linhas — inviável para OCSVM).
            if len(X_train_final) > LIMITE_AMOSTRA_OCSVM:
                try:
                    from sklearn.model_selection import train_test_split as _tts
                    X_train_svm, _, y_train_svm, _ = _tts(
                        X_train_final, y_train,
                        train_size=LIMITE_AMOSTRA_OCSVM, random_state=42, stratify=y_train,
                    )
                except ValueError:
                    # Classe minoritária pequena demais para estratificar — cai para amostra aleatória simples.
                    rng = np.random.RandomState(42)
                    idx_amostra = rng.choice(len(X_train_final), size=LIMITE_AMOSTRA_OCSVM, replace=False)
                    X_train_svm, y_train_svm = X_train_final[idx_amostra], y_train[idx_amostra]
                print(f"    (OCSVM: treino amostrado de {len(X_train_final)} para {len(X_train_svm)} linhas)")
            else:
                X_train_svm, y_train_svm = X_train_final, y_train

            modelo_svm, _, thresh_svm = anomaly_detector.optimize_one_class_svm(X_train_svm, y_train_svm)
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

    # Inclui o resumo do parse Drain3 (linhas brutas -> templates únicos) no
    # próprio JSON salvo, para não depender do console para citar o dado.
    if resumo_parse:
        resultados_finais["resumo_parse_drain3"] = resumo_parse

    # Pasta nova por execução (data/hora no nome) + salvar o JSON em disco —
    # antes só era impresso no console e se perdia (não dava para comparar
    # execuções depois, nem citar o número exato na monografia sem re-rodar).
    pasta_saida = run_output.criar_pasta_execucao("bgl_v9_polaris_tscv_iforest_vs_svm")
    caminho_json = os.path.join(pasta_saida, "resultados_bgl_v9_polaris.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(resultados_finais, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Resultado salvo em: {caminho_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação BGL via Polars (TimeSeriesSplit + iForest vs OCSVM).")
    parser.add_argument("--limite-linhas", type=int, default=None,
                         help="Limita a leitura do BGL.log às N primeiras linhas (teste rápido). "
                              "Padrão: None (lê o arquivo inteiro).")
    args = parser.parse_args()

    main(limite_linhas=args.limite_linhas)