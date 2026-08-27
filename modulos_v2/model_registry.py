"""
Persistência e recuperação de modelos treinados, por fonte de log.

O PROBLEMA QUE ISSO RESOLVE
----------------------------
No motor v6, TODO ciclo (a cada lote, de minutos em minutos) treina um
modelo do zero e o descarta assim que o ciclo termina — nada fica salvo em
disco. Duas consequências:

  1) Sem auditoria: não dá para responder "qual modelo/threshold gerou
     este resultado específico?" depois do fato — útil tanto para debugar
     quanto para citar na monografia.
  2) Sem rede de segurança: se um lote vier com dados degenerados (poucos
     logs, quase sem variação, um bug momentâneo na fonte), o motor v6
     treina e implanta esse modelo ruim de qualquer jeito — não existe
     conceito de "não confio neste modelo novo, mantenho o anterior".

Este módulo salva, a cada treino bem-sucedido (não degenerado, ver
modulos_v2/validacao_cruzada.py), um "pacote" com tudo que é necessário
para RODAR INFERÊNCIA de novo sem re-treinar: o vectorizer TF-IDF, o
modelo de redução (PCA ou SVD), o scaler das features temporais, o modelo
de detecção (IsolationForest/OneClassSVM), o threshold e os metadados
(algoritmo, redução, taxa de contaminação usada, timestamp). Quando
main_v7.py detecta que a validação cruzada do lote atual ficou
"degenerada", ele chama carregar_ultimo_modelo_bom() em vez de treinar um
modelo novo — o motor continua respondendo com o último modelo em que
efetivamente dava para confiar, em vez de implantar algo pior.

Formato em disco: joblib (mesma biblioteca que scikit-learn já usa
internamente para (de)serializar seus próprios modelos — evita puxar uma
dependência nova para o requirements.txt do projeto).
"""
import glob
import os
import time

import joblib

PASTA_BASE = "modulos_v2/model_store"


def _pasta_da_fonte(nome_fonte):
    pasta = os.path.join(PASTA_BASE, nome_fonte)
    os.makedirs(pasta, exist_ok=True)
    return pasta


def salvar_modelo_versionado(nome_fonte, pacote, manter_ultimos=10):
    """
    Salva `pacote` (dict — ver monte_pacote() em pipeline_v2.py) como um
    novo arquivo versionado por timestamp. `manter_ultimos` evita o
    diretório crescer para sempre: mantém só as N versões mais recentes por
    fonte (o restante é apagado).
    """
    pasta = _pasta_da_fonte(nome_fonte)
    carimbo = time.strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(pasta, f"modelo_{carimbo}.joblib")
    joblib.dump(pacote, caminho)

    versoes = sorted(glob.glob(os.path.join(pasta, "modelo_*.joblib")))
    for antigo in versoes[:-manter_ultimos]:
        try:
            os.remove(antigo)
        except OSError:
            pass

    return caminho


def carregar_ultimo_modelo_bom(nome_fonte):
    """
    Retorna o pacote mais recente salvo para `nome_fonte`, ou None se nunca
    houve um treino bem-sucedido persistido para essa fonte (ex.: primeira
    vez que o motor roda). Como salvar_modelo_versionado() só é chamado
    quando a CV NÃO estava degenerada, "o mais recente salvo" já é,
    por construção, "o último modelo bom".
    """
    pasta = _pasta_da_fonte(nome_fonte)
    versoes = sorted(glob.glob(os.path.join(pasta, "modelo_*.joblib")))
    if not versoes:
        return None
    return joblib.load(versoes[-1])
