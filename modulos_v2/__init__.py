"""
modulos_v2/ — extensões do motor v7 sobre a base v4/v5/v6 (pipeline.py e
modules/*.py). Nada aqui SUBSTITUI o que já existe: são módulos NOVOS,
importados apenas por main_v7.py. O motor v6 (main_v6.py) continua
funcionando exatamente como antes, sem nenhuma dependência desta pasta.

Por que uma pasta separada em vez de colocar direto em modules/:
main_v6.py, pipeline.py e todo o conteúdo de modules/ são código já
validado e citado na monografia (prints, métricas e comportamento
conhecidos). Misturar arquivos novos ali dentro criaria risco de um
import futuro (ou um "from modules import *") pegar sem querer uma
função v2 no lugar da v1, ou de uma edição futura em modules/ quebrar
v7 por engano. modulos_v2/ deixa explícito, só pelo caminho do import,
qual geração do motor está em uso.
"""
