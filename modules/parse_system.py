import os
import sys
import re
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

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


# ==========================================================================
# EXTRAÇÃO DE TIMESTAMP
# ==========================================================================
# Cada estratégia é ancorada no INÍCIO da linha (re.match, não re.search) --
# diferente da versão anterior, que buscava em qualquer parte da linha e
# podia por engano casar com um trecho parecido com data no MEIO da
# mensagem. Ancorar no início também permite CORTAR o timestamp da linha
# antes de mandar pro Drain (ver ponto 3 da explicação).
#
# Acrescentei os formatos do BGL e do HDFS (LogHub) -- testados contra os
# arquivos reais logpai/BGL/BGL_2k.log e logpai/HDFS/HDFS_2k.log: a regex
# antiga não reconhecia nenhum dos dois.
#
# Isso continua sendo uma lista de padrões conhecidos (uma "regra estática"
# no sentido estrito) -- log timestamps não têm um padrão universal
# detectável de forma 100% genérica sem ambiguidade (testei com
# dateutil.parser(fuzzy=True) e ele falha exatamente nos formatos ISO e
# BGL que aparecem nos seus próprios dados). Pra um TCC, é mais honesto
# documentar isso como uma limitação conhecida do que fingir que existe
# uma solução totalmente livre de regras aqui.
ESTRATEGIAS_TIMESTAMP = [
    # BGL/LogHub: "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ..."
    # usamos o timestamp Unix (epoch) que vem logo após o marcador de alerta
    (re.compile(r'^-?\s*(\d{10})\s+\d{4}\.\d{2}\.\d{2}\s'), 'epoch_s'),
    # HDFS/LogHub: "081109 203615 148 INFO ..." -> AAMMDD HHMMSS
    (re.compile(r'^(\d{6}\s\d{6})\s'), '%y%m%d %H%M%S'),
    # ISO 8601 / Docker: "2023-10-25T15:30:00.123Z" ou "2023-10-25 15:30:00+00:00"
    (re.compile(r'^(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)'), 'iso'),
    # Syslog/Linux: "Oct 25 15:30:00" (não inclui ano -- ver limitação abaixo)
    (re.compile(r'^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'), 'syslog'),
    # Apache: "25/Oct/2023:15:30:00"
    (re.compile(r'^(\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2})'), 'apache'),
]


def extrair_timestamp_e_corpo(line):
    """
    Tenta reconhecer um timestamp no INÍCIO da linha.
    Retorna (timestamp_bruto, formato, corpo_sem_timestamp).
    Se nenhuma estratégia bater, retorna (None, None, linha_original) --
    a linha inteira segue pro Drain, exatamente como no comportamento
    original para formatos desconhecidos.
    """
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
            # LIMITAÇÃO CONHECIDA: syslog tradicional não inclui o ano.
            # O ano retornado (padrão do pandas) NÃO é confiável -- se
            # precisar do ano correto pra essa fonte, complemente com o
            # ano do relógio local no momento da ingestão.
            return pd.to_datetime(bruto, format='%b %d %H:%M:%S', errors='coerce')
        else:  # iso
            return pd.to_datetime(bruto, format='mixed', errors='coerce')
    except (ValueError, TypeError):
        return pd.NaT


# ==========================================================================
# PARSING COM DRAIN3
# ==========================================================================
_DIRETORIO_MODULO = os.path.dirname(os.path.abspath(__file__))
CAMINHO_INI_PADRAO = os.path.join(_DIRETORIO_MODULO, "drain3.ini")
DIRETORIO_ESTADOS_PADRAO = os.path.join(_DIRETORIO_MODULO, "..", "drain3_states")


def automatic_drain_parse(file_path, nome_fonte=None,
                           caminho_ini=CAMINHO_INI_PADRAO,
                           diretorio_estados=DIRETORIO_ESTADOS_PADRAO):
    """
    Analisa os logs automaticamente usando o algoritmo Drain.

    nome_fonte: identifica qual "árvore"/estado de conhecimento reaproveitar
    entre execuções (ex: o nome da pasta de origem). Como main.py chama
    esta função com o caminho de um tempfile aleatório (via
    read_dir_to_temps), NÃO dá pra derivar isso do file_path -- por isso
    vira parâmetro explícito. Se main.py não passar nome_fonte, cai num
    estado genérico compartilhado (funciona, mas mistura o vocabulário de
    todas as fontes na mesma árvore). Ver nota no fim sobre o ajuste de
    uma linha necessário em main.py para separar por fonte.
    """
    caminho_completo = os.path.abspath(file_path)

    # ----- Config: carrega máscaras (IP, hex, UUID, números) do drain3.ini -----
    config = TemplateMinerConfig()
    if os.path.isfile(caminho_ini):
        config.load(caminho_ini)
    else:
        print(f"⚠️  '{caminho_ini}' não encontrado — rodando SEM máscaras "
              f"(IPs, hex, UUIDs e números não serão normalizados antes do "
              f"clustering, o que tende a multiplicar clusters desnecessariamente).")

    # ----- Persistência: reaproveita a árvore de templates entre execuções -----
    if nome_fonte is None:
        nome_fonte = "generico"
        print("⚠️  nome_fonte não informado — usando estado compartilhado "
              "'generico'. Para manter o histórico de templates separado por "
              "fonte de log, chame automatic_drain_parse(path, nome_fonte=<pasta>).")

    os.makedirs(diretorio_estados, exist_ok=True)
    caminho_estado = os.path.join(diretorio_estados, f"drain3_state_{nome_fonte}.bin")
    persistencia = FilePersistence(caminho_estado)

    template_miner = TemplateMiner(persistence_handler=persistencia, config=config)

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Pula linhas vazias

            # Extrai o timestamp e CORTA ele da linha antes de mandar pro Drain
            # (recomendação oficial do projeto Drain3: mandar só a parte
            # livre da mensagem melhora a acurácia do parsing).
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

    if not data:
        print("Nenhum log encontrado ou processado.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['Timestamp'] = df.apply(
        lambda r: _converter_timestamp(r['Timestamp'], r['Timestamp_Formato']), axis=1
    )
    df = df.drop(columns=['Timestamp_Formato'])

    total_clusters = df['Cluster_ID'].nunique()
    print(f"Arquivo '{caminho_completo}' lido! Encontradas {len(df)} linhas. "
          f"Extraídos {df['Timestamp'].notna().sum()} timestamps. "
          f"Clusters (acumulado da fonte '{nome_fonte}'): {total_clusters}")

    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python parse_system.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]

    if not os.path.isfile(log_file):
        print(f"Erro: O arquivo '{log_file}' não existe.")
        sys.exit(1)

    df = automatic_drain_parse(log_file, nome_fonte=os.path.basename(log_file))

    print("\nAmostra dos dados extraídos:")
    print(df[['File', 'Cluster_ID', 'Template']].head())  # 'Arquivo' não existe -- era 'File'
