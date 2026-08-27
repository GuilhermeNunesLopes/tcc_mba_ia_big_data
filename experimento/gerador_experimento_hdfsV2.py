import os
import time
import pandas as pd
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
    print(f"\n[2/4] Lendo logs brutos e extraindo templates com parse_system.py...")
    
    # Aciona a sua função original que já carrega o drain3.ini com as máscaras corretas
    parsed_generator = automatic_drain_parse(log_path, nome_fonte="HDFS_TCC")
    df_parsed = pd.DataFrame(parsed_generator, ignore_index=True)
    # O parse_system retorna o dataframe com a coluna 'Raw_Log'. 
    # Vamos extrair o BlockId direto dela usando Regex.
    print("   -> Extraindo BlockIds estruturais das mensagens brutas...")
    df_parsed['BlockId'] = df_parsed['Raw_Log'].str.extract(r'(blk_-?\d+)')
    
    # Limpa linhas que não pertencem a nenhum bloco
    df_parsed = df_parsed.dropna(subset=['BlockId'])
    
    # Padroniza a nomenclatura (O parse_system chama de Cluster_ID, precisamos como EventId para agrupar)
    df_parsed = df_parsed.rename(columns={'Cluster_ID': 'EventId'})
    
    print(f"   -> {len(df_parsed)} linhas vinculadas a blocos com sucesso.")

    # ---------------------------------------------------------
    # 3. CONSTRUÇÃO DA MATRIZ DE FREQUÊNCIA 
    # ---------------------------------------------------------
    print(f"\n[3/4] Agrupando matriz de eventos por BlockId...")
    
    df_grouped = df_parsed.groupby(['BlockId', 'EventId']).size().unstack(fill_value=0).reset_index()
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