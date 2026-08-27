import os
import json
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence

# Nome da sua pasta específica
pasta_binarios = "../drain3_states"  
arquivo_saida = "../templates_drain_consolidados.json"

dados_consolidados = {}

# Verifica se a pasta existe no diretório atual
if not os.path.exists(pasta_binarios):
    print(f"Erro: A pasta '{pasta_binarios}' não foi encontrada no diretório atual.")
    exit()

# Lista e processa os arquivos dentro de 'drain3_states'
arquivos = os.listdir(pasta_binarios)
arquivos_bin = [arq for arq in arquivos if arq.endswith(".bin")]

print(f"Encontrados {len(arquivos_bin)} arquivos binários na pasta '{pasta_binarios}'.\n")

for nome_arquivo in arquivos_bin:
    caminho_completo = os.path.join(pasta_binarios, nome_arquivo)
    print(f"Lendo: {nome_arquivo}...")
    
    try:
        persistence = FilePersistence(caminho_completo)
        template_miner = TemplateMiner(persistence)
        
        templates_do_arquivo = []
        for cluster in template_miner.drain.clusters:
            templates_do_arquivo = [
                *templates_do_arquivo,
                {
                    "id_cluster": cluster.cluster_id,
                    "frequencia": cluster.size,
                    "template": cluster.get_template()
                }
            ]
        
        # Usa o nome do arquivo como chave para organizar o JSON
        dados_consolidados[nome_arquivo] = templates_do_arquivo
        
    except Exception as e:
        print(f"Erro ao processar {nome_arquivo}: {e}")

# Salva o resultado final formatado
with open(arquivo_saida, "w", encoding="utf-8") as f:
    json.dump(dados_consolidados, f, indent=4, ensure_ascii=False)

print(f"\nSucesso! Arquivo gerado com os dados de todos os estados: {arquivo_saida}")