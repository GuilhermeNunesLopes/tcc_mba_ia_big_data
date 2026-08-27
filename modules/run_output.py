"""
Helper único para pastas de saída por execução.

Problema que resolve: cada script de experimento escrevia direto em
"resultados/arquivo.html", então toda vez que alguém rodava o mesmo script
de novo (com outro parâmetro, outro dia, para comparar) o resultado anterior
era sobrescrito silenciosamente. `experimento_pipeline_graficos.py` já tinha
uma tentativa de pasta por dia (`pasta_saida = f"resultados/{data_hoje}"`),
mas usava concatenação de string sem separador ao montar os caminhos dos
arquivos (`f"{pasta_saida}grafico_x.html"`), o que colava a data no nome do
arquivo em vez de criar uma subpasta de verdade — ver achado corrigido junto
com esta adição.

Uso: cada script chama UMA VEZ, no início, `criar_pasta_execucao("tag_do_experimento")`
e usa o caminho retornado (com os.path.join) para TODOS os arquivos que gerar
nessa execução (parquet, json, html, png).
"""
import os
import time


def criar_pasta_execucao(tag, base="resultados"):
    """
    Cria e retorna 'resultados/<AAAAMMDD_HHMMSS>_<tag>/' — uma subpasta nova
    a cada execução, carimbada com data e hora, para permitir comparar
    execuções diferentes do mesmo script (ou do mesmo script com parâmetros
    diferentes) ao longo do tempo, sem que uma sobrescreva a saída da outra.
    """
    carimbo = time.strftime("%Y%m%d_%H%M%S")
    pasta = os.path.join(base, f"{carimbo}_{tag}")
    os.makedirs(pasta, exist_ok=True)
    return pasta
