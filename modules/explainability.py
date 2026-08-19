"""
Módulo de Explicabilidade (XAI) para o Detector de Anomalias em Logs.

O modelo (Isolation Forest / OCSVM) decide em cima de features reduzidas
(PCA/SVD), que são combinações lineares de milhares de termos TF-IDF —
por isso não dá para simplesmente perguntar ao modelo "qual termo pesou
mais". A estratégia adotada aqui é mais simples e mais robusta para efeitos
de explicação a um humano: para cada log marcado como anômalo, mostramos
os termos de MAIOR PESO TF-IDF presentes naquele log específico (ou seja,
os termos mais raros/característicos daquele log em relação ao vocabulário
aprendido no treino). Isso não explica literalmente a decisão matemática do
Isolation Forest, mas dá um indício textual interpretável do que torna
aquele log incomum — que é o suficiente para uma triagem operacional e para
justificar o alerta numa apresentação.
"""

import numpy as np
import scipy.sparse as sp


def explicar_log_individual(linha_tfidf, feature_names, top_n=5):
    """
    Recebe UMA linha (esparsa ou densa) da matriz TF-IDF e retorna uma string
    com os termos de maior peso naquele log, no formato "termo (peso)".

    Termos com peso 0 (ausentes do log) nunca aparecem na explicação, mesmo
    que top_n seja maior que a quantidade de termos presentes.
    """
    if sp.issparse(linha_tfidf):
        linha = linha_tfidf.toarray().ravel()
    else:
        linha = np.asarray(linha_tfidf).ravel()

    if not np.any(linha):
        return "sem termos distintivos"

    indices_ordenados = np.argsort(linha)[::-1]
    indices_top = [i for i in indices_ordenados[:top_n] if linha[i] > 0]

    termos = [f"{feature_names[i]} ({linha[i]:.2f})" for i in indices_top]
    return ", ".join(termos)


def explicar_anomalias(tfidf_matrix, vectorizer, posicoes_anomalas, top_n=5):
    """
    Gera a explicação textual (top termos TF-IDF) apenas para as posições
    marcadas como anômalas — evita gastar tempo com o restante do batch,
    que pode ter dezenas de milhares de logs normais.

    Parâmetros
    ----------
    tfidf_matrix : matriz esparsa TF-IDF do conjunto (ex.: tfidf_test),
        na MESMA ordem posicional (0-based) em que X_test foi construído.
    vectorizer : TfidfVectorizer já ajustado (para pegar o vocabulário).
    posicoes_anomalas : array de posições (0-based) da matriz TF-IDF que
        correspondem aos logs marcados como anomalia. Use a coluna
        '_row_pos' de df_resultado para obter essas posições corretamente,
        mesmo depois do DataFrame ter sido reordenado por anomaly_score.
    top_n : quantidade máxima de termos retornados por log.

    Retorna
    -------
    dict {posicao (int): explicação (str)} — para você atribuir de volta
    ao df_resultado usando as mesmas posições de '_row_pos'.
    """
    feature_names = vectorizer.get_feature_names_out()
    explicacoes = {}

    for pos in posicoes_anomalas:
        explicacoes[int(pos)] = explicar_log_individual(
            tfidf_matrix[pos], feature_names, top_n=top_n
        )

    return explicacoes