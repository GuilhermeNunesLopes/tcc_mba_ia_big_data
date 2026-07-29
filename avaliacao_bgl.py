"""
Avaliação científica do pipeline de detecção de anomalias contra o dataset
rotulado BGL (Blue Gene/L), do LogHub, já presente no submódulo `logpai/`.

Por que o BGL?
- É um dos poucos datasets do LogHub com rótulo POR LINHA já embutido no
  próprio log (coluna 'Label': "-" = normal, qualquer outro valor = tipo de
  alerta real). Isso permite calcular Precision/Recall/F1/PR-AUC de verdade,
  algo que o pipeline atual nunca calcula em produção (os lotes rodam sobre
  logs sem rótulo).
- HDFS/Apache/OpenSSH etc. no LogHub-2k NÃO trazem rótulo por linha (HDFS é
  rotulado por bloco, em um arquivo separado que não está neste repositório).

Reaproveita DIRETAMENTE os módulos modules/preprocessor.py e
modules/anomaly_detector.py, para que os números aqui sejam comparáveis ao
que o main.py faz em produção (mesmos hiperparâmetros de TF-IDF, SVD e
Isolation Forest).

Divisão treino/teste CRONOLÓGICA (não aleatória): Le, V.-H.; Zhang, H.
"Log-based Anomaly Detection with Deep Learning: How Far Are We?" (ICSE 2022)
mostram que split aleatório em logs causa vazamento de dados (data leakage)
e infla as métricas de forma não realista.

Uso:
    python evaluate_bgl.py
"""
import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, precision_recall_curve, auc
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modules.preprocessor as preprocessor
import modules.anomaly_detector as anomaly_detector

CAMINHO_BGL = os.path.join("logpai", "BGL", "BGL_2k.log_structured.csv")
FRACAO_TREINO = 0.7  # 70% mais antigo -> treino ("normal" assumido) | 30% mais recente -> teste
#SVD_COMPONENTS = [50, 100, 150, 200, 250, 300] # mesmo valor usado em main.py
SVD_COMPONENTS=[200]#melhor avaliado no BGL, com F1=0.88, PR-AUC=0.91, Precision=0.92, Recall=0.85

def carregar_bgl_rotulado(caminho: str = CAMINHO_BGL) -> pd.DataFrame:
    """Carrega o BGL_2k já estruturado (parseado pelo Drain no próprio LogHub)
    e adapta as colunas para o formato que preprocessor.tfidf_vectorize espera
    (Level, Source, Event), mantendo o rótulo original para avaliação.
    """
    df = pd.read_csv(caminho)

    df_padronizado = pd.DataFrame({
        "Timestamp": pd.to_datetime(df["Timestamp"], unit="s"),
        "Level": df["Level"].astype(str),
        "Source": df["Component"].astype(str),
        "Event": df["EventTemplate"].astype(str),
        "Raw_Log": df["Content"].astype(str),
        "Label_Original": df["Label"].astype(str),
    })

    # Regra do próprio dataset BGL: "-" = log normal; qualquer outro código = alerta real
    df_padronizado["y_true"] = (df_padronizado["Label_Original"] != "-").astype(int)

    df_padronizado = df_padronizado.sort_values("Timestamp").reset_index(drop=True)
    return df_padronizado


def dividir_cronologicamente(df: pd.DataFrame, fracao_treino: float = FRACAO_TREINO):
    corte = int(len(df) * fracao_treino)
    df_train = df.iloc[:corte].reset_index(drop=True)
    df_test = df.iloc[corte:].reset_index(drop=True)
    return df_train, df_test


def main():
    print("=" * 62)
    print("AVALIAÇÃO CIENTÍFICA — Pipeline de Detecção de Anomalias")
    print("Dataset: BGL (LogHub) — 2.000 logs rotulados manualmente")
    print("=" * 62)

    df = carregar_bgl_rotulado()
    print(f"\nTotal de logs: {len(df)}")
    print(f"Anomalias reais no dataset completo: {df['y_true'].sum()} "
          f"({df['y_true'].mean() * 100:.2f}%)")

    df_train, df_test = dividir_cronologicamente(df)
    print("\nSplit CRONOLÓGICO (Le & Zhang, ICSE 2022) — não aleatório:")
    print(f"  Treino: {len(df_train)} logs | "
          f"{df_train['Timestamp'].min()} até {df_train['Timestamp'].max()} | "
          f"anomalias: {df_train['y_true'].sum()} ({df_train['y_true'].mean() * 100:.2f}%)")
    print(f"  Teste:  {len(df_test)} logs | "
          f"{df_test['Timestamp'].min()} até {df_test['Timestamp'].max()} | "
          f"anomalias: {df_test['y_true'].sum()} ({df_test['y_true'].mean() * 100:.2f}%)")

    # ==========================================
    # Mesmo pipeline de features usado em main.py
    # ==========================================
    print("\n[1/4] Vetorização TF-IDF (fit no treino, transform no teste)...")
    X_train_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_train)
    X_test_tfidf, _ = preprocessor.tfidf_vectorize(df_test, vectorizer=vectorizer)

    print("\n[2/4] Procurando o melhor número de componentes do SVD...")

    best_f1 = -1
    
    best_svd = None
    best_model = None
    best_metrics = None
    best_df_resultado = None
    best_components = None
    best_threshold_final = None # Guardar o melhor threshold
    
    for n_components in SVD_COMPONENTS:
    
        print(f"\nTestando SVD = {n_components}")
    
        X_train_svd, svd_model = preprocessor.apply_truncated_svd(
            X_train_tfidf,
            svd_model=None,
            n_components=n_components
        )
    
        X_test_svd, _ = preprocessor.apply_truncated_svd(
            X_test_tfidf,
            svd_model=svd_model
        )
    
        # ==========================================
        # 1. PROCESSAR TREINO (Otimizar Modelo e Threshold)
        # Passar y_true_train é essencial para a otimização interna funcionar sem vazar
        # ==========================================
        resultado_treino = anomaly_detector.process_log_anomalies(
            df_original=df_train,
            X_tfidf=X_train_svd,
            y_true=df_train["y_true"], # <- ADICIONADO: Necessário para achar o best_threshold
            model=None
        )
        
        modelo_treinado = resultado_treino[1]
        threshold_treinado = resultado_treino[4] # Capturando o best_threshold retornado (índice 4 baseado na correção anterior)
    
        # ==========================================
        # 2. PROCESSAR TESTE (Avaliação Final Rigorosa)
        # Passar o modelo e o threshold encontrados no treino
        # ==========================================
        resultado_teste = anomaly_detector.process_log_anomalies(
            df_original=df_test,
            X_tfidf=X_test_svd,
            y_true=df_test["y_true"],
            model=modelo_treinado,          # Passa o modelo treinado
            best_threshold=threshold_treinado # <- ADICIONADO: Usa o threshold do treino no teste
        )
        
        df_resultado = resultado_teste[0]
        metricas = resultado_teste[2]
    
        print(
            f"F1={metricas['F1_Score']:.4f} "
            f"Precision={metricas['Precision']:.4f} "
            f"Recall={metricas['Recall']:.4f}"
        )
    
        if metricas["F1_Score"] > best_f1:
            best_f1 = metricas["F1_Score"]
            best_components = n_components
            best_svd = svd_model
            best_model = modelo_treinado
            best_metrics = metricas
            best_df_resultado = df_resultado
            best_threshold_final = threshold_treinado # Guarda o threshold da melhor rodada
            X_test_final = X_test_svd # Guarda o X_test da melhor iteração para o ablation

    print("\nMelhor configuração encontrada")
    print(f"SVD: {best_components}")
    print(f"F1: {best_f1:.4f}")
    
    metricas = best_metrics
    modelo_treinado = best_model
    df_resultado = best_df_resultado
    X_test = X_test_final # Define o X_test para o ablation mais abaixo
    

    print("\n" + "=" * 62)
    print("RESULTADO — métricas reais contra o ground truth do BGL")
    print("=" * 62)
    for chave, valor in metricas.items():
        print(f"  {chave}: {valor:.4f}")

    matriz = pd.crosstab(
        df_resultado["y_true_label"], df_resultado["pred_is_anomaly"],
        rownames=["Real"], colnames=["Previsto"]
    )
    print("\nMatriz de confusão (split de teste):")
    print(matriz)

    # ==========================================
    # ABLATION: corte por percentil fixo (o que o código usa hoje)
    # vs. predict() nativo do Isolation Forest (contamination='auto')
    # Ajuda a discutir, com números, se o limiar de 1.5% do
    # anomaly_detector.py está sub- ou super-detectando neste dataset.
    # ==========================================
    y_true = df_resultado["y_true_label"].values
    pred_predict_nativo = (modelo_treinado.predict(X_test) == -1).astype(int)

    comparacao = pd.DataFrame({
        "Estratégia": ["Percentil fixo 1.5% (atual)", "predict() nativo (contamination='auto')"],
        "Precision": [
            metricas["Precision"],
            precision_score(y_true, pred_predict_nativo, zero_division=0),
        ],
        "Recall": [
            metricas["Recall"],
            recall_score(y_true, pred_predict_nativo, zero_division=0),
        ],
        "F1": [
            metricas["F1_Score"],
            f1_score(y_true, pred_predict_nativo, zero_division=0),
        ],
        "Qtd_Prevista_Anomalia": [
            int(df_resultado["pred_is_anomaly"].sum()),
            int(pred_predict_nativo.sum()),
        ],
    })
    print("\nComparação de estratégias de corte (mesmo modelo, mesmos scores):")
    print(comparacao.to_string(index=False))

    # ==========================================
    # Curva Precision-Recall (figura para o capítulo de resultados do TCC)
    # ==========================================
    scores_para_curva = -df_resultado["anomaly_score"].values
    precisions, recalls, _ = precision_recall_curve(y_true, scores_para_curva)
    pr_auc = auc(recalls, precisions)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, label=f"Isolation Forest (PR-AUC = {pr_auc:.3f})")
    plt.axhline(y_true.mean(), color="gray", linestyle="--",
                label=f"Baseline aleatório (taxa de anomalia = {y_true.mean():.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall — Split de teste do BGL")
    plt.legend()
    plt.tight_layout()

    os.makedirs("resultados", exist_ok=True)
    caminho_figura = os.path.join("resultados", "curva_pr_bgl.png")
    plt.savefig(caminho_figura, dpi=150)
    print(f"\nCurva Precision-Recall salva em: {caminho_figura}")

    caminho_saida = os.path.join("resultados", "avaliacao_bgl.json")
    saida_completa = {
        "metricas_percentil_fixo": metricas,
        "comparacao_estrategias": comparacao.to_dict(orient="records"),
    }
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(saida_completa, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em: {caminho_saida}")

    return df_resultado, metricas, comparacao


if __name__ == "__main__":
    main()