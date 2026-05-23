import os
import sys
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# Mantive sua função de diretório intacta, caso ainda precise dela
import tempfile
def read_dir_to_temps(directory):
    temp_files = []
    if not os.path.isdir(directory):
        print("Directory not found.")
        return []

    for nome_arquivo in os.listdir(directory):
        caminho_completo = os.path.join(directory, nome_arquivo)
        if os.path.isfile(caminho_completo):
            with open(caminho_completo, 'r', encoding='utf-8') as arquivo:
                conteudo = arquivo.read()

            fp = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
            fp.write(conteudo)
            fp.close() 
            temp_files.append(fp.name)
            
    return temp_files

def automatic_drain_parse(file_path):
    """
    Analisa os logs automaticamente usando o algoritmo Drain.
    Não precisa de Regex hardcoded.
    """
    # Configuração padrão do Drain3 (já vem com máscaras para IPs, Números, etc.)
    config = TemplateMinerConfig()
    template_miner = TemplateMiner(config=config)
    
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue # Pula linhas vazias
            
            # O Drain processa a linha e tenta encontrar a qual "cluster/template" ela pertence
            result = template_miner.add_log_message(line)
            
            # result contém o ID do template, o template em si e os parâmetros extraídos
            data.append({
                'Raw_Log': line,
                'Cluster_ID': result["cluster_id"],
                'Template': result["template_mined"],
                'Parameters': result.get("parameters", [])
            })
            
    if not data:
        print("Nenhum log encontrado ou processado.")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Exibe um resumo de quantos templates únicos foram encontrados
    total_clusters = df['Cluster_ID'].nunique()
    print(f"Encontradas {len(df)} linhas de log, agrupadas em {total_clusters} templates (clusters).")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python parse_system.py <log_file>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    
    if not os.path.isfile(log_file):
        print(f"Erro: O arquivo '{log_file}' não existe.")
        sys.exit(1)
        
    df = automatic_drain_parse(log_file)
    
    # Exibe as primeiras linhas do dataframe resultante
    print("\nAmostra dos dados extraídos:")
    print(df[['Cluster_ID', 'Template']].head())