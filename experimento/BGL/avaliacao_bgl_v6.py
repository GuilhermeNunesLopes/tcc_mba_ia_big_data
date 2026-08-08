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
import modules.parse_system as parse_system
import numpy as np
import tempfile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.sparse as sp





CAMINHO_BGL = os.path.join("logpai", "BGL", "BGL_2k.log_structured.csv")
FRACAO_TREINO = 0.7  # 70% mais antigo -> treino ("normal" assumido) | 30% mais recente -> teste
SVD_COMPONENTS = [15, 30, 50] # mesmo valor usado em main.py
#SVD_COMPONENTS = [2, 5, 10]
#SVD_COMPONENTS = [50, 100, 150]
#SVD_COMPONENTS=[300]#melhor avaliado no BGL, com F1=0.88, PR-AUC=0.91, Precision=0.92, Recall=0.85

def carregar_bgl_rotulado(caminho: str = CAMINHO_BGL) -> pd.DataFrame:
    print("\n[0/4] Acionando parse_system.py para extrair templates na mosca...")
    df = pd.read_csv(caminho)

    # 1. Cria um arquivo temporário com o texto cru (Content)
    # Como a sua função exige um file_path físico, criamos um na memória
    fd, temp_path = tempfile.mkstemp(text=True, suffix=".log")
    with open(fd, 'w', encoding='utf-8') as f:
        for linha in df["Content"]:
            # Escreve apenas o corpo da mensagem
            f.write(str(linha) + "\n")

    # 2. Chama a SUA função de parse nativa
    print("Processando arquivo temporário com o Drain3...")
    df_parsed = parse_system.automatic_drain_parse(
        file_path=temp_path,
        nome_fonte="BGL_Eval" # Passa o nome para criar o state separado, conforme sua função exige
    )
    
    # 3. Limpa o arquivo temporário do sistema operacional
    os.remove(temp_path)

    # Verifica se o parse retornou a mesma quantidade de linhas
    if len(df_parsed) != len(df):
        raise ValueError("Ocorreu um erro: o parse_system devolveu um número diferente de linhas.")

    # 4. Substitui o template antigo do LogHub pelo template gerado pelo SEU código
    df_padronizado = pd.DataFrame({
        "Timestamp": pd.to_datetime(df["Timestamp"], unit="s"),
        "Level": df["Level"].astype(str),
        "Source": df["Component"].astype(str),
        "Event": df_parsed["Template"].astype(str),  # <--- O template limpo pelo seu Drain3
        
        # Mantemos a Técnica 1: Concatenar Level (Severidade) + Evento limpo
        "Raw_Log": df["Level"].astype(str) + " " + df_parsed["Template"].astype(str),
        
        "Label_Original": df["Label"].astype(str),
    })

    # Criação do Ground Truth (y_true)
    df_padronizado["y_true"] = (df_padronizado["Label_Original"] != "-").astype(int)
    
    # Ordenação cronológica
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
    print("Abordagem: Linha a Linha + Engenharia de Features Temporais")
    print("=" * 62)

    df_bruto = carregar_bgl_rotulado()
    
    # ========================================================
    # TÉCNICA MASTER: INJEÇÃO DE CONTEXTO TEMPORAL
    # Calculamos antes do split para não perder dados na borda do teste
    # ========================================================
    print("\nCalculando features temporais (Log Burst e Time Diff)...")
    # 1. Segundos desde o último log
    df_bruto['time_diff'] = df_bruto['Timestamp'].diff().dt.total_seconds().fillna(0)
    
    # 2. Rajada de logs (quantos logs ocorreram nos últimos 60 segundos)
    df_bruto_idx = df_bruto.set_index('Timestamp')
    df_bruto['rolling_count'] = df_bruto_idx.index.to_series().rolling('60s').count().values
    
    # Faz o split
    df_train, df_test = dividir_cronologicamente(df_bruto)
    
    print("\nSplit CRONOLÓGICO:")
    print(f"  Treino: {len(df_train)} logs | anomalias: {df_train['y_true'].sum()}")
    print(f"  Teste:  {len(df_test)} logs | anomalias: {df_test['y_true'].sum()}")

    # ==========================================
    # Vetorização TF-IDF
    # ==========================================
    print("\n[1/4] Vetorização TF-IDF (fit no treino, transform no teste)...")
    X_train_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_train)
    X_test_tfidf, _ = preprocessor.tfidf_vectorize(df_test, vectorizer=vectorizer)

    print("\n[2/4] Fundindo Matriz Esparsa com Metadados (Físicos e Temporais)...")
    from sklearn.preprocessing import MinMaxScaler
    import scipy.sparse as sp

    # TÉCNICA: Injeção de Metadados Físicos e Severidade
    df_train['log_length'] = df_train['Raw_Log'].apply(len)
    df_test['log_length'] = df_test['Raw_Log'].apply(len)

    error_levels = ['FATAL', 'ERROR', 'SEVERE', 'WARN', 'WARNING', 'CRITICAL']
    df_train['is_error'] = df_train['Level'].str.upper().isin(error_levels).astype(float)
    df_test['is_error'] = df_test['Level'].str.upper().isin(error_levels).astype(float)

    # Pegamos as 4 features extras: tamanho, se_é_erro, tempo_desde_ultimo, contagem_por_minuto
    cols_meta = ['log_length', 'is_error', 'time_diff', 'rolling_count']
    
    scaler_meta = MinMaxScaler()
    meta_train = scaler_meta.fit_transform(df_train[cols_meta])
    meta_test = scaler_meta.transform(df_test[cols_meta])

    # FUSÃO: TF-IDF Esparso Direto + Nossas 4 Features Poderosas
    X_train_final = sp.hstack((X_train_tfidf, meta_train), format='csr')
    X_test_final = sp.hstack((X_test_tfidf, meta_test), format='csr')

    # ==========================================
    # 1. PROCESSAR TREINO 
    # ==========================================
    print("\nFase de Treino: Otimizando Isolation Forest...")
    resultado_treino = anomaly_detector.process_log_anomalies(
        df_original=df_train,
        X_tfidf=X_train_final,
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
        X_tfidf=X_test_final,
        y_true=df_test["y_true"],
        model=modelo_treinado,          
        best_threshold=threshold_treinado 
    )
    
    df_resultado = resultado_teste[0]
    metricas = resultado_teste[2]

    print("\nMelhor configuração encontrada")
    print(f"F1: {metricas['F1_Score']:.4f}")
    
    X_test = X_test_final 
    
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

    y_true = df_resultado["y_true_label"].values
    pred_predict_nativo = (modelo_treinado.predict(X_test) == -1).astype(int)

    comparacao = pd.DataFrame({
        "Estratégia": ["Threshold Otimizado", "predict() nativo (contamination='auto')"],
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