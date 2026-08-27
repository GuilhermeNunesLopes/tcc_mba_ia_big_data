"""
Escrita atômica de parquet/JSON para o motor v7.

Por que isso é uma melhoria de robustez:
main_v6.py escreve direto no caminho final —
df_resultado.to_parquet(caminho_parquet) e json.dump(obj, open(caminho, "w")).
Enquanto isso, modules/dashboard.py está rodando em paralelo (outro
processo) e pode tentar LER esse mesmo arquivo a qualquer momento —
inclusive no meio da escrita. O risco concreto:

  1) Se o motor for interrompido (Ctrl+C, queda de energia, o processo
     Python travar) bem no meio de um to_parquet()/json.dump(), o arquivo
     fica truncado/corrompido no disco. Na PRÓXIMA vez que o dashboard
     tentar ler esse arquivo (inclusive minutos ou horas depois, sem
     nenhuma relação com a falha original), a leitura quebra com um erro
     de parsing — um bug "fantasma" bem difícil de associar à causa raiz
     real (a escrita interrompida há tempos).
  2) Mesmo sem interrupção, escrever é uma operação não-instantânea (o
     arquivo cresce aos poucos no disco); uma leitura que aconteça bem
     nesse meio-tempo pode pegar um arquivo parcialmente escrito.

A técnica clássica para eliminar os dois riscos é: escrever em um arquivo
TEMPORÁRIO (no mesmo diretório, para garantir que fique no mesmo
filesystem) e, só quando a escrita terminar com sucesso, renomear
(os.replace) para o nome final. os.replace é atômico no nível do sistema
operacional — quem for ler o arquivo final SEMPRE vai ver ou a versão
antiga completa, ou a versão nova completa, nunca uma mistura ou um
arquivo pela metade.
"""
import json
import os
import tempfile


def escrever_parquet_atomico(df, caminho_final):
    """Salva um DataFrame em parquet de forma atômica (ver docstring do módulo)."""
    pasta = os.path.dirname(os.path.abspath(caminho_final)) or "."
    os.makedirs(pasta, exist_ok=True)

    fd, caminho_temp = tempfile.mkstemp(suffix=".parquet.tmp", dir=pasta)
    os.close(fd)
    try:
        df.to_parquet(caminho_temp, index=False)
        os.replace(caminho_temp, caminho_final)
    except Exception:
        if os.path.exists(caminho_temp):
            os.remove(caminho_temp)
        raise


def escrever_json_atomico(objeto, caminho_final, indent=None, ensure_ascii=True):
    """Salva um objeto como JSON de forma atômica (ver docstring do módulo)."""
    pasta = os.path.dirname(os.path.abspath(caminho_final)) or "."
    os.makedirs(pasta, exist_ok=True)

    fd, caminho_temp = tempfile.mkstemp(suffix=".json.tmp", dir=pasta)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(objeto, f, indent=indent, ensure_ascii=ensure_ascii)
        os.replace(caminho_temp, caminho_final)
    except Exception:
        if os.path.exists(caminho_temp):
            os.remove(caminho_temp)
        raise
