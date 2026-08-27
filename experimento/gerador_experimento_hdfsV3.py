import os
import time
import pandas as pd
import polars as pl
from sklearn.utils import shuffle
import sys
from pathlib import Path
# ==============================================================================
# IMPORTAÇÃO DOS MÓDULOS DA ARQUITETURA
# ==============================================================================
# Reaproveita os scripts já validados na pasta 'modules'

sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.dowloand_dataset_hugging import download_dataset_file
from modules.parse_system import automatic_drain_parse


def preparar_experimento_hdfs(
    log_path='experimento/HDFS.log', 
    label_path='experimento/anomaly_label.csv', 
    output_dir='resultados'
):
    print("="*60)
    print("🚀 INICIANDO PREPARAÇÃO DO DATASET HDFS COM MÓDULOS")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. DOWNLOAD AUTOMÁTICO VIA HUGGING FACE
    # ---------------------------------------------------------
    if not os.path.exists(log_path):
        print(f"\n[1/4] '{log_path}' não encontrado. Acionando módulo de download...")
        # Utiliza sua função do dowloand_dataset_hugging.py
        log_path = download_dataset_file(repo_id="logpai/loghub", filename="HDFS/HDFS_2k.log", local_dir=".")
        
    if not os.path.exists(label_path):
        print(f"Baixando arquivo de labels...")
        label_path = download_dataset_file(repo_id="logpai/loghub", filename="HDFS/anomaly_label.csv", local_dir=".")

    # ---------------------------------------------------------
    # 2. PARSING COM O SEU SISTEMA DRAIN (parse_system.py)
    # ---------------------------------------------------------
    print(f"\n[2/4] e [3/4] Processando lotes com POLARS (Extração e Agrupamento)...")
    
    lotes_agregados = []
    
    # Itera sobre os pedaços (DataFrames Pandas) gerados pelo automatic_drain_parse
    for df_chunk_pd in automatic_drain_parse(log_path, nome_fonte="HDFS_TCC"):
        # 1. Converte o pedaço do Pandas para Polars (rápido e eficiente)
        df_pl = pl.from_pandas(df_chunk_pd)
        
        # 2. Encadeamento otimizado do Polars: extrai regex, filtra, renomeia e conta
        contagem_lote = (
            df_pl
            # Extrai o primeiro grupo de captura (1) da Regex
            .with_columns(pl.col("Raw_Log").str.extract(r"(blk_-?\d+)", 1).alias("BlockId"))
            .drop_nulls(subset=["BlockId"])
            .rename({"Cluster_ID": "EventId"})
            .group_by(["BlockId", "EventId"])
            .len(name="Count") # Conta as ocorrências
        )
        lotes_agregados.append(contagem_lote)

    print("   -> Lotes processados. Consolidando a matriz final...")

    # 3. Junta todos os lotes agregados (verticalmente)
    df_todas_contagens = pl.concat(lotes_agregados)
    
    # 4. Soma blocos que apareceram em lotes diferentes e faz o Pivot (equivalente ao unstack)
    df_matriz_pl = (
        df_todas_contagens
        .group_by(["BlockId", "EventId"])
        .agg(pl.col("Count").sum())
        .pivot(index="BlockId", on="EventId", values="Count")
        .fill_null(0) # Preenche os NaNs com 0
    )
    
    # 5. Converte de volta para Pandas para não quebrar a etapa 4 do seu script
    df_grouped = df_matriz_pl.to_pandas()
    
    # 6. Padroniza os nomes das colunas de eventos (adiciona o "E")
    df_grouped.columns = ['BlockId'] + [f'E{col}' for col in df_grouped.columns if col != 'BlockId']
    # ---------------------------------------------------------
    # 4. ROTULAÇÃO E SPLIT 50/50 (UNDERSAMPLING)
    # ---------------------------------------------------------
    print(f"\n[4/4] Aplicando Labels e separando Treino vs Teste (50/50)...")
    
    labels = pd.read_csv(label_path)
    labels['Label'] = labels['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
    
    df_final = pd.merge(df_grouped, labels, on='BlockId', how='inner')
    
    blocos_anomalos = df_final[df_final['Label'] == 1]
    blocos_normais = df_final[df_final['Label'] == 0]
    
    # Força a proporção exata de 50%
    blocos_normais_teste = blocos_normais.sample(n=len(blocos_anomalos), random_state=42)
    dataset_teste_50_50 = pd.concat([blocos_anomalos, blocos_normais_teste])
    
    # Embaralha para que os algoritmos de detecção avaliem sem viés de ordem
    dataset_teste_50_50 = shuffle(dataset_teste_50_50, random_state=42).reset_index(drop=True)
    
    # O restante vira ambiente saudável (100% normal) para aprendizado não supervisionado
    dataset_treino_normal = blocos_normais.drop(blocos_normais_teste.index).reset_index(drop=True)

    # Exportando em Parquet para garantir leitura super-rápida no dashboard
    treino_path = os.path.join(output_dir, 'hdfs_treino_normal.parquet')
    teste_path = os.path.join(output_dir, 'hdfs_teste_50_50.parquet')
    
    dataset_treino_normal.to_parquet(treino_path, index=False)
    dataset_teste_50_50.to_parquet(teste_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Concluído em {elapsed:.2f} segundos!")
    print(f"Dataset de Treino (Puro): {len(dataset_treino_normal)} blocos.")
    print(f"Dataset de Teste (50/50): {len(dataset_teste_50_50)} blocos.")
    print(f"Arquivos salvos em:\n - {treino_path}\n - {teste_path}")

if __name__ == '__main__':
    preparar_experimento_hdfs()