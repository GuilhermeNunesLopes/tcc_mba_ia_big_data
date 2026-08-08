import os
import re
import time
import pandas as pd
from sklearn.utils import shuffle
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# ==============================================================================
# PIPELINE DE PREPARAÇÃO DO DATASET HDFS PARA O TCC
# Estrutura a extração do Drain, separa os blocos e cria o cenário balanceado
# Foco exclusivo em clusterização não supervisionada e correlação de grafos
# ==============================================================================

def preparar_experimento_hdfs(
    log_path='HDFS.log', 
    label_path='anomaly_label.csv', 
    output_dir='resultados'
):
    print("="*60)
    print("🚀 INICIANDO PREPARAÇÃO DO DATASET HDFS (TREINO vs TESTE 50/50)")
    print("="*60)

    if not os.path.exists(log_path) or not os.path.exists(label_path):
        print(f"\n⚠️ ERRO: Arquivos de origem não encontrados.")
        print(f"Certifique-se de que '{log_path}' e '{label_path}' estão na mesma pasta do script.")
        print("Dica: Você pode usar a função de download do Hugging Face para obter os arquivos.\n")
        return

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. PARSING COM DRAIN3 E EXTRAÇÃO DE BLOCKID
    # ---------------------------------------------------------
    print(f"\n[1/4] Lendo logs brutos e extraindo templates com Drain3...")
    config = TemplateMinerConfig()
    miner = TemplateMiner(config=config)
    
    parsed_data = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Extrai o BlockId usando expressão regular (Padrão HDFS: blk_-1608999687919862906)
            block_match = re.search(r'(blk_-?\d+)', line)
            block_id = block_match.group(1) if block_match else None
            
            if block_id:
                # O Drain processa o texto livre e devolve o ID do cluster (EventId)
                result = miner.add_log_message(line)
                parsed_data.append({
                    'BlockId': block_id,
                    'EventId': result['cluster_id']
                })
                
    df_parsed = pd.DataFrame(parsed_data)
    print(f"   -> {len(df_parsed)} linhas processadas.")
    print(f"   -> Total de Templates (Eventos Únicos) mapeados: {df_parsed['EventId'].nunique()}")

    # ---------------------------------------------------------
    # 2. CONSTRUÇÃO DA MATRIZ DE FREQUÊNCIA (PREPARAÇÃO TF-IDF)
    # ---------------------------------------------------------
    print(f"\n[2/4] Agrupando eventos por BlockId (Matriz de Contagem)...")
    
    # Conta quantas vezes cada EventId ocorreu dentro de um mesmo BlockId
    df_grouped = df_parsed.groupby(['BlockId', 'EventId']).size().unstack(fill_value=0).reset_index()
    
    # Renomeia as colunas para o formato 'E1', 'E2', etc.
    df_grouped.columns = ['BlockId'] + [f'E{col}' for col in df_grouped.columns if col != 'BlockId']
    print(f"   -> Matriz base criada. Total de blocos únicos: {len(df_grouped)}")

    # ---------------------------------------------------------
    # 3. ROTULAÇÃO E SPLIT DOS DADOS
    # ---------------------------------------------------------
    print(f"\n[3/4] Aplicando Labels e separando Treino (Normal) vs Teste (50/50)...")
    
    labels = pd.read_csv(label_path)
    # Padroniza para formato binário (1=Anomaly, 0=Normal)
    labels['Label'] = labels['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
    
    # Junta as labels na matriz agrupada
    df_final = pd.merge(df_grouped, labels, on='BlockId', how='inner')
    
    blocos_anomalos = df_final[df_final['Label'] == 1]
    blocos_normais = df_final[df_final['Label'] == 0]
    
    print(f"   -> Identificados {len(blocos_anomalos)} blocos anômalos reais.")
    print(f"   -> Identificados {len(blocos_normais)} blocos normais reais.")

    # ---------------------------------------------------------
    # 4. UNDERSAMPLING E GERAÇÃO DOS ARQUIVOS DE EXPERIMENTO
    # ---------------------------------------------------------
    print(f"\n[4/4] Realizando undersampling para 50/50...")
    
    # Sorteia blocos normais na mesma quantidade exata de anomalias
    blocos_normais_teste = blocos_normais.sample(n=len(blocos_anomalos), random_state=42)
    
    # Constrói o dataset balanceado para avaliação (Isolation Forest / Matriz de Confusão)
    dataset_teste_50_50 = pd.concat([blocos_anomalos, blocos_normais_teste])
    dataset_teste_50_50 = shuffle(dataset_teste_50_50, random_state=42).reset_index(drop=True)
    
    # Constrói o dataset de treino usando apenas o que sobrou dos logs normais
    # Isso garante que a base do modelo será o comportamento natural do sistema
    dataset_treino_normal = blocos_normais.drop(blocos_normais_teste.index).reset_index(drop=True)

    print(f"   -> Dataset Treino gerado: {dataset_treino_normal.shape[0]} blocos (100% Normais)")
    print(f"   -> Dataset Teste gerado:  {dataset_teste_50_50.shape[0]} blocos (50% Anomalias / 50% Normais)")

    # Salvando em Parquet para otimizar o carregamento no Streamlit e economizar espaço
    treino_path = os.path.join(output_dir, 'hdfs_treino_normal.parquet')
    teste_path = os.path.join(output_dir, 'hdfs_teste_50_50.parquet')
    
    dataset_treino_normal.to_parquet(treino_path, index=False)
    dataset_teste_50_50.to_parquet(teste_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Concluído em {elapsed:.2f} segundos!")
    print(f"Arquivos salvos em:\n - {treino_path}\n - {teste_path}")
    print("\nAgora você pode aplicar o Vectorizer (TF-IDF) nestes dataframes e alimentar os algoritmos de detecção.")

if __name__ == '__main__':
    # Quando tiver os arquivos HDFS.log e anomaly_label.csv na mesma pasta, basta rodar o script.
    preparar_experimento_hdfs(
        log_path='HDFS.log', 
        label_path='anomaly_label.csv', 
        output_dir='resultados'
    )
