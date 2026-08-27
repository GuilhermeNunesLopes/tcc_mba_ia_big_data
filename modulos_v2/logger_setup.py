"""
Logging estruturado para o motor v7.

Por que isso é uma melhoria de robustez (e não só "cosmético"):
main_v6.py só usa print(). Isso funciona bem enquanto alguém está
olhando o terminal, mas o motor roda em loop por horas/dias — se ele
travar às 3h da manhã, o print já rolou para fora do buffer do
terminal (ou o terminal nem existe mais, se rodou via serviço/agendador)
e não sobra nenhum rastro do que aconteceu. Este módulo grava as MESMAS
mensagens (o texto e os emojis continuam iguais, para quem está
acompanhando ao vivo no terminal) só que TAMBÉM em um arquivo de log
rotativo em disco — então um problema que aconteceu de madrugada pode
ser diagnosticado de manhã.

RotatingFileHandler evita o log crescer para sempre: quando
logs_motor/motor_v7.log passa de MAX_BYTES, ele vira
motor_v7.log.1, .2, etc., até BACKUP_COUNT arquivos — depois disso os
mais antigos são descartados automaticamente.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

PASTA_LOGS = "logs_motor"
ARQUIVO_LOG = os.path.join(PASTA_LOGS, "motor_v7.log")
MAX_BYTES = 5 * 1024 * 1024  # 5 MB por arquivo
BACKUP_COUNT = 5             # mantém até 5 arquivos antigos (motor_v7.log.1 .. .5)


def configurar_logger(nome="motor_v7"):
    """
    Cria (ou reaproveita, se já configurado) um logger que escreve tanto no
    console (mesmo formato enxuto que main_v6.py já usa) quanto num arquivo
    rotativo em disco. Chamar mais de uma vez é seguro — não duplica handlers.
    """
    os.makedirs(PASTA_LOGS, exist_ok=True)
    logger = logging.getLogger(nome)

    if logger.handlers:
        return logger  # já configurado (ex.: chamado de novo no mesmo processo)

    logger.setLevel(logging.INFO)

    formato_arquivo = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    formato_console = logging.Formatter("%(message)s")  # mantém o visual do v6 (sem timestamp/nível)

    handler_arquivo = RotatingFileHandler(
        ARQUIVO_LOG, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler_arquivo.setFormatter(formato_arquivo)
    handler_arquivo.setLevel(logging.INFO)

    handler_console = logging.StreamHandler()
    handler_console.setFormatter(formato_console)
    handler_console.setLevel(logging.INFO)

    logger.addHandler(handler_arquivo)
    logger.addHandler(handler_console)
    logger.propagate = False

    return logger
