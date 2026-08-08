import os
import sys
import re
import tempfile
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

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


# ==========================================================================
# EXTRAÇÃO DE TIMESTAMP
# ==========================================================================
ESTRATEGIAS_TIMESTAMP = [
    (re.compile(r'^-?\s*(\d{10})\s+\d{4}\.\d{2}\.\d{2}\s'), 'epoch_s'),
    (re.compile(r'^(\d{6}\s\d{6})\s'), '%y%m%d %H%M%S'),
    (re.compile(r'^(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)'), 'iso'),
    (re.compile(r'^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'), 'syslog'),
    (re.compile(r'^(\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2})'), 'apache'),
]

def extrair_timestamp_e_corpo(line):
    for regex, formato in ESTRATEGIAS_TIMESTAMP:
        m = regex.match(line)
        if m:
            return m.group(1), formato, line[m.end():].strip()
    return None, None, line

def _converter_timestamp(bruto, formato):
    if bruto is None:
        return pd.NaT
    try:
        if formato == 'epoch_s':
            return pd.to_datetime(int(bruto), unit='s')
        elif formato == '%y%m%d %H%M%S':
            return pd.to_datetime(bruto, format=formato)
        elif formato == 'apache':
            return pd.to_datetime(bruto, format='%d/%b/%Y:%H:%M:%S')
        elif formato == 'syslog':
            return pd.to_datetime(bruto, format='%b %d %H:%M:%S', errors='coerce')
        else:  
            return pd.to_datetime(bruto, format='mixed', errors='coerce')
    except (ValueError, TypeError):
        return pd.NaT


# ==========================================================================
# PARSING COM DRAIN3 (PROCESSAMENTO EM LOTES)
# ==========================================================================
_DIRETORIO_MODULO = os.path.dirname(os.path.abspath(__file__))
CAMINHO_INI_PADRAO = os.path.join(_DIRETORIO_MODULO, "drain3.ini")
DIRETORIO_ESTADOS_PADRAO = os.path.join(_DIRETORIO_MODULO, "..", "drain3_states")


def _processar_dataframe_lote(data_lote):
    """Função auxiliar para converter a lista de dicionários em DataFrame."""
    df = pd.DataFrame(data_lote)
    if not df.empty:
        df['Timestamp'] = df.apply(
            lambda r: _converter_timestamp(r['Timestamp'], r['Timestamp_Formato']), axis=1
        )
        df = df.drop(columns=['Timestamp_Formato'])
    return df


def automatic_drain_parse(file_path, nome_fonte=None,
                           caminho_ini=CAMINHO_INI_PADRAO,
                           diretorio_estados=DIRETORIO_ESTADOS_PADRAO,
                           tamanho_lote=100000):
    """
    Analisa os logs automaticamente usando o algoritmo Drain.
    Transformado em um gerador (yield) para processar arquivos gigantes sem estourar a RAM.
    """
    caminho_completo = os.path.abspath(file_path)

    config = TemplateMinerConfig()
    if os.path.isfile(caminho_ini):
        config.load(caminho_ini)
    else:
        print(f"⚠️  '{caminho_ini}' não encontrado — rodando SEM máscaras.")

    if nome_fonte is None:
        nome_fonte = "generico"
        print("⚠️  nome_fonte não informado — usando estado compartilhado 'generico'.")

    os.makedirs(diretorio_estados, exist_ok=True)
    caminho_estado = os.path.join(diretorio_estados, f"drain3_state_{nome_fonte}.bin")
    persistencia = FilePersistence(caminho_estado)

    template_miner = TemplateMiner(persistence_handler=persistencia, config=config)

    data = []
    linhas_processadas = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue 

            timestamp_bruto, formato_ts, corpo_para_drain = extrair_timestamp_e_corpo(line)
            result = template_miner.add_log_message(corpo_para_drain)

            data.append({
                'File': caminho_completo,
                'Timestamp': timestamp_bruto,
                'Timestamp_Formato': formato_ts,
                'Raw_Log': line,
                'Cluster_ID': result["cluster_id"],
                'Template': result["template_mined"],
                'Parameters': result.get("parameters", [])
            })

            # Quando atingir o tamanho do lote, processa, entrega (yield) e limpa a memória
            if len(data) >= tamanho_lote:
                df = _processar_dataframe_lote(data)
                linhas_processadas += len(df)
                total_clusters = df['Cluster_ID'].nunique()
                print(f"Lote processado! Linhas até agora: {linhas_processadas}. "
                      f"Clusters no lote: {total_clusters}")
                
                yield df
                data = [] # Esvazia a lista para liberar RAM

    # Processa qualquer resíduo que tenha sobrado no último lote
    if data:
        df = _processar_dataframe_lote(data)
        linhas_processadas += len(df)
        total_clusters = df['Cluster_ID'].nunique()
        print(f"Último lote processado! Linhas totais: {linhas_processadas}. "
              f"Clusters no lote: {total_clusters}")
        yield df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python parse_system.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]

    if not os.path.isfile(log_file):
        print(f"Erro: O arquivo '{log_file}' não existe.")
        sys.exit(1)

    nome_da_fonte = os.path.basename(log_file)
    
    # Como a função agora usa yield, iteramos sobre os lotes gerados
    # Tamanho do lote de 100.000 é um bom balanço entre velocidade e consumo de memória
    gerador_lotes = automatic_drain_parse(log_file, nome_fonte=nome_da_fonte, tamanho_lote=100000)

    for i, df_lote in enumerate(gerador_lotes):
        print(f"\nAmostra do Lote {i + 1}:")
        print(df_lote[['File', 'Cluster_ID', 'Template']].head())
        
        # Aqui você pode salvar cada lote em disco se precisar
        # df_lote.to_parquet(f"log_processado_lote_{i}.parquet")