import sys
import os
import time
import json
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

# ==============================================================================
# CONFIGURAÇÃO DE ROTA PARA A PASTA 'MODULES'
# ==============================================================================
pasta_atual = os.path.dirname(os.path.abspath(__file__))
pasta_pai = os.path.abspath(os.path.join(pasta_atual, '..'))
if pasta_pai not in sys.path:
    sys.path.insert(0, pasta_pai)

# ==============================================================================
# IMPORTAÇÃO DOS MÓDULOS DA ARQUITETURA
# ==============================================================================
from modules.anomaly_detector import process_log_anomalies
from modules.preprocessor import apply_truncated_svd
from modules.parse_system import automatic_drain_parse
import modules.visualizer as visualizer
import modules.run_output as run_output

def executar_experimento_completo_literatura():
    print("="*60)
    print("🚀 INICIANDO EXPERIMENTO COMPLETO - LITERATURA (100% DOS DADOS HDFS)")
    print("="*60)
    
    log_path = 'experimento/test2/HDFS_42k.log'
    #log_path = 'experimento/test1/HDFS_23k.log'
    #log_path = 'HDFS.log'
    label_path = 'experimento/anomaly_label.csv'
    # Pasta nova por execução (data/hora no nome) — evita que uma rodada
    # sobrescreva a evidência da anterior, permitindo comparar ao longo do tempo.
    output_dir = run_output.criar_pasta_execucao("literatura_v2")
    start_time = time.time()

    if not os.path.exists(log_path) or not os.path.exists(label_path):
        print(f"⚠️ ERRO: '{log_path}' ou '{label_path}' não encontrados na pasta atual.")
        return

    # ---------------------------------------------------------
    # 1. PARSING E AGRUPAMENTO ESCALÁVEL (PARA ARQUIVOS GIGANTES)
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] [1/4] Extraindo templates e agrupando blocos em lotes...")
    
    resumo_parse = {}
    batch_log = automatic_drain_parse(log_path, nome_fonte="HDFS_Completo", tamanho_lote=100000,
                                       resumo_saida=resumo_parse)

    lista_frequencias_lotes = []

    # O "for" volta para garantir que a RAM não estoure
    for i, df_lote in enumerate(batch_log):
        # 1. Extrai BlockId e limpa
        df_lote['BlockId'] = df_lote['Raw_Log'].str.extract(r'(blk_-?\d+)')
        df_lote = df_lote.dropna(subset=['BlockId'])
        df_lote = df_lote.rename(columns={'Cluster_ID': 'EventId'})
        
        # 2. Conta as ocorrências dos eventos APENAS neste lote (MUITO leve na RAM)
        # O reset_index com name='Count' cria uma tabela com: BlockId | EventId | Count
        freq_lote = df_lote.groupby(['BlockId', 'EventId']).size().reset_index(name='Count')
        
        # 3. Guarda a contagem numérica e a função descarta o texto pesado do lote (Raw_Log)
        lista_frequencias_lotes.append(freq_lote)
        print(f"   -> Lote {i+1} agregado e texto descartado da RAM.")

    # Salva o resumo do parse (linhas brutas -> templates únicos, ver
    # modules/parse_system.py) junto dos resultados desta execução, em vez de
    # só imprimir no console e perder ao fechar o terminal.
    if resumo_parse:
        caminho_resumo_parse = os.path.join(output_dir, "resumo_parse_drain3.json")
        with open(caminho_resumo_parse, "w", encoding="utf-8") as f:
            json.dump(resumo_parse, f, indent=2, ensure_ascii=False)
        print(f"   -> Resumo do parse salvo em: {caminho_resumo_parse}")

    # 4. Fora do loop: Junta apenas as matrizes numéricas e soma os blocos que
    # eventualmente começaram em um lote e terminaram em outro.
    print(f"\n[{time.strftime('%H:%M:%S')}] Consolidando frequências globais...")
    df_frequencias_totais = pd.concat(lista_frequencias_lotes, ignore_index=True)
    
    # Soma as contagens e cria a matriz final (BlockId nas linhas, EventId nas colunas)
    df_grouped = df_frequencias_totais.groupby(['BlockId', 'EventId'])['Count'].sum().unstack(fill_value=0).reset_index()
    df_grouped.columns = ['BlockId'] + [f'E{col}' for col in df_grouped.columns if col != 'BlockId']

    print(f"   -> Matriz estruturada final com {len(df_grouped)} blocos totais.")

    # ---------------------------------------------------------
    # 2. APLICAÇÃO DO GROUND TRUTH
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] [2/4] Cruzando dados reais com o anomaly_label.csv...")
    labels = pd.read_csv(label_path)
    labels['Label'] = labels['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
    
    df_final = pd.merge(df_grouped, labels, on='BlockId', how='inner')
    X_raw = df_final.drop(columns=['BlockId', 'Label'])
    y_true = df_final['Label']

    taxa_real_anomalia = (y_true.sum() / len(y_true))
    print(f"   -> Total avaliado: {len(df_final)} blocos.")
    print(f"   -> Anomalias reais detectadas no label: {y_true.sum()} ({taxa_real_anomalia:.2%})")

    # ---------------------------------------------------------
    # 3. PRÉ-PROCESSAMENTO (TF-IDF + SVD)
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] [3/4] Aplicando vetorização e compressão SVD...")
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_raw)
    X_reduzido, svd_model = apply_truncated_svd(X_tfidf, n_components=100)

    # ---------------------------------------------------------
    # 4. TREINAMENTO E INFERÊNCIA DO MODELO
    # ---------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] [4/4] Executando detecção via Isolation Forest...")
    
    # O process_log_anomalies otimiza internamente com F1-Score usando 20% para validação cruzada
    # Passamos os labels originais para que ele tente encontrar o melhor threshold sem vazar dados.
    df_resultado, modelo_treinado, metricas, best_params, threshold = process_log_anomalies(
        df_original=df_final, 
        X_tfidf=X_reduzido, 
        y_true=y_true, 
        algorithm="iforest"
    )

        # ---------------------------------------------------------
    # 5. MÉTRICAS ACADÊMICAS (sem reotimizar o threshold aqui)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📊 RESULTADOS ACADÊMICOS DO EXPERIMENTO (100% BASE)")
    print("="*50)

    # 1. Score contínuo — usado só para a curva PR-AUC (métrica de ranking,
    #    não depende de um corte escolhido, então calculá-la sobre o dataset
    #    todo não é vazamento)
    scores_decision = -modelo_treinado.decision_function(X_reduzido)

    # 2. Curva Precision-Recall e PR-AUC
    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_true, scores_decision)
    pr_auc = auc(recalls_curve, precisions_curve)

    # 3. Classificação binária: NÃO recalculamos o threshold aqui.
    #    df_resultado['pred_is_anomaly'] já veio de process_log_anomalies,
    #    aplicando o threshold calibrado no split interno de validação
    #    (dentro de optimize_isolation_forest). Recalcular o corte usando
    #    y_true diretamente aqui seria escolher o threshold olhando a
    #    resposta certa do "teste" — vazamento de dados.

    # 4. Métricas finais com o corte já calibrado sem vazamento
    #    IMPORTANTE: process_log_anomalies devolve df_resultado já reordenado
    #    por anomaly_score (sort_values no fim da função) — comparar essa
    #    coluna 'pred_is_anomaly' (na ordem nova) contra y_true (que continua
    #    na ordem original de df_final) compara cada linha com o rótulo de
    #    OUTRA linha. Usamos df_resultado['y_true_label'], que foi atribuída
    #    ANTES do sort_values (portanto se moveu junto com cada linha) e por
    #    isso fica corretamente alinhada com 'pred_is_anomaly'.
    prec = precision_score(df_resultado['y_true_label'], df_resultado['pred_is_anomaly'], zero_division=0)
    rec = recall_score(df_resultado['y_true_label'], df_resultado['pred_is_anomaly'], zero_division=0)
    f1 = f1_score(df_resultado['y_true_label'], df_resultado['pred_is_anomaly'], zero_division=0)

    print(f"Limiar (Threshold) calibrado (validação interna): {threshold:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    # ==========================================================================
    # 6. GERAÇÃO DE EVIDÊNCIAS GRÁFICAS
    # ==========================================================================
    print(f"\n[{time.strftime('%H:%M:%S')}] 📈 Gerando evidências visuais...")
    
    df_resultado['Timestamp'] = df_resultado.index
    
    event_cols = [c for c in df_resultado.columns if str(c).startswith('E')]
    def criar_pseudo_template(row):
        eventos = [col for col in event_cols if row[col] > 0]
        return "Bloco: " + " | ".join(eventos)
    df_resultado['Template'] = df_resultado.apply(criar_pseudo_template, axis=1)

    fig_dist = visualizer.plot_anomaly_distribution_plotly(df_resultado)
    fig_dist.write_html(os.path.join(output_dir, "grafico_distribuicao_completo_literaturav2.html"))

    fig_timeline = visualizer.plot_anomaly_timeline_plotly(df_resultado)
    fig_timeline.write_html(os.path.join(output_dir, "grafico_timeline_completo_literaturav2.html"))

    # Novos Gráficos: Comparativo e Métricas
    fig_comparativo = visualizer.plot_comparativo_antes_depois(df_resultado)
    fig_comparativo.write_html(os.path.join(output_dir, "grafico_comparativo_completo_literatura.html"))

    # Passando prec, rec e f1 (nomes usados neste script)
    fig_metricas = visualizer.plot_metricas_destaque(prec, rec, f1)
    fig_metricas.write_html(os.path.join(output_dir, "grafico_metricas_completo_literatura.html"))

    # Curva Precision-Recall + sensibilidade ao threshold: gerada aqui, pelo
    # próprio script, a partir de scores_decision/y_true/threshold já
    # calculados acima (passo 5) — não é um cálculo feito à parte, é a
    # mesma evidência usada para justificar o PR-AUC impresso no console.
    visualizer.plot_pr_curve_threshold(
        y_true=y_true,
        scores=scores_decision,
        threshold_usado=threshold,
        titulo="Literatura v2 — HDFS 100% da amostra rotulada (Drain3 + TF-IDF + SVD + Isolation Forest)",
        caminho_saida=os.path.join(output_dir, "pr_curve_literatura_v2.png"),
    )

    print(f"[{time.strftime('%H:%M:%S')}] ✅ Execução finalizada! Gráficos salvos em: {output_dir}")
    print(f"Tempo total: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    executar_experimento_completo_literatura()
