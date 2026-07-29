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
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler






CAMINHO_BGL = os.path.join("logpai", "BGL", "BGL_2k.log_structured.csv")
FRACAO_TREINO = 0.7  # 70% mais antigo -> treino ("normal" assumido) | 30% mais recente -> teste
SVD_COMPONENTS = [15, 30, 50] # mesmo valor usado em main.py
#SVD_COMPONENTS = [2, 5, 10]
#SVD_COMPONENTS = [50, 100, 150]
#SVD_COMPONENTS=[300]#melhor avaliado no BGL, com F1=0.88, PR-AUC=0.91, Precision=0.92, Recall=0.85

def carregar_bgl_rotulado(caminho: str = CAMINHO_BGL) -> pd.DataFrame:
    df = pd.read_csv(caminho)

    df_padronizado = pd.DataFrame({
        "Timestamp": pd.to_datetime(df["Timestamp"], unit="s"),
        "Level": df["Level"].astype(str),
        "Source": df["Component"].astype(str),
        "Event": df["EventTemplate"].astype(str),
        
        # O SEGREDO AQUI: Substituir o texto cru (Content) pelo template limpo!
        # Isso reduz a dimensionalidade do TF-IDF drasticamente e foca no padrão do erro.
        "Raw_Log": df["EventTemplate"].astype(str), 
        
        "Label_Original": df["Label"].astype(str),
    })

    df_padronizado["y_true"] = (df_padronizado["Label_Original"] != "-").astype(int)
    df_padronizado = df_padronizado.sort_values("Timestamp").reset_index(drop=True)
    return df_padronizado

def agrupar_por_janela_de_tempo(df: pd.DataFrame, tamanho_janela: str = '1min') -> pd.DataFrame:
    """
    Agrupa logs individuais em blocos (janelas) de tempo.
    tamanho_janela: '1min', '5min', '30S' (segundos), etc.
    """
    print(f"\nAgrupando logs em janelas de {tamanho_janela}...")
    
    # Garante que o dataframe está ordenado cronologicamente
    df = df.sort_values("Timestamp")
    
    # Agrupa usando o Grouper do Pandas baseado na coluna de tempo
    agrupado = df.groupby(pd.Grouper(key='Timestamp', freq=tamanho_janela))
    
    dados_janela = []
    
    for timestamp_janela, grupo in agrupado:
        if len(grupo) == 0:
            continue # Pula janelas vazias de tempo onde nenhum log ocorreu
            
        # Transforma todos os templates de log (Event) da janela em uma única "frase"
        # Onde cada template funciona como uma "palavra" para o TF-IDF
        sequencia_eventos = " ".join(grupo['Event'].astype(str).tolist())
        
        # Label da Janela: 1 (Anomalia) se PELO MENOS UM log na janela for anômalo
        janela_anomala = 1 if grupo['y_true'].sum() > 0 else 0
        
        dados_janela.append({
            "Timestamp": timestamp_janela,
            "Event": sequencia_eventos,      # Sobrescrevemos para o seu TF-IDF ler
            "Raw_Log": sequencia_eventos,    # Sobrescrevemos para o seu TF-IDF ler
            "y_true": janela_anomala,
            "Qtd_Logs_Na_Janela": len(grupo)
        })
        
    df_janelas = pd.DataFrame(dados_janela)
    
    print(f"Redução: de {len(df)} logs individuais para {len(df_janelas)} blocos de tempo.")
    print(f"Janelas anômalas no total: {df_janelas['y_true'].sum()} "
          f"({df_janelas['y_true'].mean()*100:.2f}%)")
          
    return df_janelas
def dividir_cronologicamente(df: pd.DataFrame, fracao_treino: float = FRACAO_TREINO):
    corte = int(len(df) * fracao_treino)
    df_train = df.iloc[:corte].reset_index(drop=True)
    df_test = df.iloc[corte:].reset_index(drop=True)
    return df_train, df_test


def main():
    print("=" * 62)
    print("AVALIAÇÃO CIENTÍFICA — Pipeline de Detecção de Anomalias")
    print("Abordagem: Janelas de Tempo (Time Windows)")
    print("=" * 62)

    # 1. Carrega os logs originais linha a linha
    df_bruto = carregar_bgl_rotulado()
    
    # ========================================================
    # 2. A MÁGICA ACONTECE AQUI: Transforma linhas em Janelas
    # Como BGL_2k é um dataset pequeno (amostra de 2 mil linhas), 
    # usar janelas de '30S' (segundos) ou '1min' é ideal para não 
    # ficarmos com poucos dados de treino.
    # ========================================================
    #df_janelas = agrupar_por_janela_de_tempo(df_bruto, tamanho_janela='1min')
    df_janelas = agrupar_por_janela_de_tempo(df_bruto, tamanho_janela='5S') # 5 Segundos
    # 3. Faz o split cronológico SOBRE AS JANELAS (e não sobre as linhas)
    #df_train, df_test = dividir_cronologicamente(df_janelas)
    
    #print("\nSplit CRONOLÓGICO por Janelas:")
    #print(f"  Treino: {len(df_train)} janelas | anomalias: {df_train['y_true'].sum()}")
    #print(f"  Teste:  {len(df_test)} janelas | anomalias: {df_test['y_true'].sum()}")


    # ==========================================
    # Mesmo pipeline de features usado em main.py
    # ==========================================
    print("\n[1/4] Vetorização TF-IDF (fit no treino, transform no teste)...")
    X_train_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_train)
    X_test_tfidf, _ = preprocessor.tfidf_vectorize(df_test, vectorizer=vectorizer)

    print("\n[2/4] Preparando matrizes (SVD e Scalers foram removidos)...")
    # Passamos a matriz esparsa bruta diretamente para o modelo
    X_train_final = X_train_tfidf
    X_test_final = X_test_tfidf

    # ==========================================
    # 1. PROCESSAR TREINO 
    # ==========================================
    print("\nFase de Treino: Otimizando Isolation Forest...")
    resultado_treino = anomaly_detector.process_log_anomalies(
        df_original=df_train,
        X_tfidf=X_train_final,   # <--- Usando a matriz esparsa bruta do TF-IDF
        y_true=df_train["y_true"], 
        model=None
    )
    
    modelo_treinado = resultado_treino[1]
    threshold_treinado = resultado_treino[4] 

    # ==========================================
    # 2. PROCESSAR TESTE 
    # ==========================================
    print("\nFase de Teste: Avaliando métricas reais...")
    resultado_teste = anomaly_detector.process_log_anomalies(
        df_original=df_test,
        X_tfidf=X_test_final,    # <--- Usando a matriz esparsa bruta do TF-IDF
        y_true=df_test["y_true"],
        model=modelo_treinado,          
        best_threshold=threshold_treinado 
    )
    
    df_resultado = resultado_teste[0]
    metricas = resultado_teste[2]

    print("\nMelhor configuração encontrada")
    print(f"F1: {metricas['F1_Score']:.4f}")
    
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