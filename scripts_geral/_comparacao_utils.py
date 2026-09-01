"""
_comparacao_utils.py — carregamento compartilhado dos resultados de
avaliacao_walkforward.py para os scripts gera-comparacao-monografia*.py.

CORREÇÃO (27/08/2026): antes desta correção, cada script varria uma pasta de
resultados e montava as séries a comparar sem checar se duas pastas com o
MESMO padrão de nome (ex.: dois "walkforward_iforest_pca_n20_t4pct") eram de
fato comparáveis. Na prática duas pastas assim podiam conter execuções sobre
datasets completamente diferentes — ex.: BGL.log completo (4.747.963 linhas,
rodado no PC do usuário) vs. BGL_filtrado.log (17.258 linhas, rodado numa
sessão do Claude) — e os gráficos anteriores misturavam essas execuções como
se fossem a mesma coisa, sem nenhum aviso (ver Historico-2026-08-26.md,
entrada de 27/08 "revalidação dos resultados").

Este módulo centraliza a leitura + deduplicação para os 3 scripts de
comparação, para que a correção valha para todos e não precise ser mantida
em 3 lugares separados.
"""
import os
import re
import json


# Execuções cujos números já foram publicados em
# Resultados_Discussao_TCC_Comparativo_2026-08-26.docx (Seção 3.1) — usadas
# como critério de desempate em carregar_experimentos_walkforward() para que
# os gráficos nunca divirjam silenciosamente do texto já entregue. Validado
# em 27/08/2026 (ver Historico-2026-08-26.md, "revalidação dos resultados").
PASTAS_CANONICAS = {
    '20260825_233045_walkforward_iforest_pca_n5_t15pct',  # IForest/PCA, F1=0,1186
    '20260826_221220_walkforward_iforest_svd_n5_t15pct',  # IForest/SVD, F1=0,1064
    '20260825_233147_walkforward_ocsvm_pca_n5_t15pct',    # OCSVM/PCA,   F1=0,0173
    '20260826_221322_walkforward_ocsvm_svd_n5_t15pct',    # OCSVM/SVD,   F1=0,0858 (Rodada B, Seção 3.1.1)
}


def extrair_timestamp_pasta(caminho_arquivo):
    """Extrai um timestamp ordenável (string 'YYYYMMDD_HHMMSS') do nome da
    pasta que contém o JSON, a partir do padrão <YYYYMMDD>_<HHMMSS>_... usado
    pelo avaliacao_walkforward.py/avaliacao_walkforwardv2.py. Retorna '' (o
    menor valor possível na ordenação, ou seja, "mais antigo") se o nome da
    pasta não casar esse padrão — ex.: pastas antigas com nomes livres como
    '20260825_15/walkforward_5_split'.
    """
    nome_pasta = os.path.basename(os.path.dirname(caminho_arquivo))
    m = re.match(r'^(\d{8})_(\d{6})', nome_pasta)
    return f"{m.group(1)}_{m.group(2)}" if m else ''


def carregar_experimentos_walkforward(pasta_alvo, pastas_ignoradas=None):
    """Varre pasta_alvo (recursivamente) por avaliacao_walkforward.json
    válidos (com as chaves 'resumo' e 'config') e devolve uma lista de dicts,
    UM POR COMBINAÇÃO ÚNICA de (algoritmo, redução, n_splits, test_size,
    dataset). "dataset" aqui é o total de linhas brutas lidas
    (resumo_parse_drain3), não só o nome da pasta — é isso que detecta duas
    execuções com nome parecido mas sobre arquivos de log diferentes.

    Quando existe mais de uma execução para a mesma combinação exata (ex.:
    reexecução após correção de bug), mantém só a MAIS RECENTE (pelo
    timestamp no nome da pasta) e imprime um aviso dizendo qual foi
    descartada e por quê — nunca descarta em silêncio.

    Cada item devolvido tem: dados (json completo), config, algoritmo,
    reducao, n_splits, test_size, linhas_dataset, chave, timestamp,
    nome_pasta, arquivo.
    """
    pastas_ignoradas = set(pastas_ignoradas or [])

    arquivos_json = []
    for root, dirs, files in os.walk(pasta_alvo):
        dirs[:] = [d for d in dirs if d not in pastas_ignoradas]
        for file in files:
            if file == 'avaliacao_walkforward.json':
                arquivos_json.append(os.path.join(root, file))

    brutos = []
    for arquivo in arquivos_json:
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
        except Exception as e:
            print(f" -> Erro ao ler {arquivo}: {e}")
            continue

        if 'resumo' not in dados or 'config' not in dados:
            continue

        config = dados['config']
        algoritmo = str(config.get('algoritmo', 'Alg')).upper()
        reducao = str(config.get('reducao', 'Sem_Reducao')).upper()
        n_splits = config.get('n_splits', 'N/A')
        test_size = config.get('test_size', 'N/A')

        # Fingerprint do dataset: soma de linhas brutas lidas pelo Drain3.
        # É isso que separa "BGL.log completo" de "BGL_filtrado.log" mesmo
        # quando n_splits/test_size/algoritmo/redução são idênticos.
        resumo_parse = dados.get('resumo_parse_drain3')
        linhas_dataset = None
        if isinstance(resumo_parse, list) and resumo_parse:
            try:
                linhas_dataset = sum(
                    item.get('linhas_arquivo_bruto', 0)
                    for item in resumo_parse if isinstance(item, dict)
                )
            except TypeError:
                linhas_dataset = None
        if not linhas_dataset:
            linhas_dataset = 'desconhecido'

        chave = (algoritmo, reducao, n_splits, test_size, linhas_dataset)
        brutos.append({
            'arquivo': arquivo,
            'dados': dados,
            'config': config,
            'algoritmo': algoritmo,
            'reducao': reducao,
            'n_splits': n_splits,
            'test_size': test_size,
            'linhas_dataset': linhas_dataset,
            'chave': chave,
            'timestamp': extrair_timestamp_pasta(arquivo),
            'nome_pasta': os.path.basename(os.path.dirname(arquivo)),
        })

    # Deduplicação: para cada chave exata, mantém só UMA execução.
    #
    # Regra de desempate:
    #  1) Se alguma das execuções concorrentes está em PASTAS_CANONICAS (as
    #     execuções cujos números já foram publicados em
    #     Resultados_Discussao_TCC_Comparativo_2026-08-26.docx, Seção 3.1),
    #     essa vence — mesmo que não seja a mais recente. Sem essa regra, uma
    #     reexecução incidental (ex.: os testes de tempo de execução PCA vs.
    #     SVD, que rodam o MESMO experimento várias vezes só para medir
    #     tempo) acaba "vencendo" por ser mais recente, e o gráfico passa a
    #     mostrar um F1 diferente do que está escrito no TCC — uma
    #     inconsistência nova, do mesmo tipo que esta correção pretende
    #     eliminar. (Achado real: 20260827_162321_walkforward_iforest_pca_n5_t15pct,
    #     um dos 4 reruns de medição de tempo, deu F1=0,1420 — diferente dos
    #     F1=0,1186 já publicados a partir de 20260825_233045.)
    #  2) Caso contrário (nenhuma das concorrentes é uma pasta canônica),
    #     mantém a execução mais recente pelo timestamp no nome da pasta —
    #     apropriado para resolver duplicatas que são claramente reruns
    #     obsoletos (ex.: execuções de antes da correção do vazamento de
    #     estado do Drain3, ver pr_curve_evidencia_visual na memória do
    #     projeto).
    melhores = {}
    for item in brutos:
        chave = item['chave']
        atual = melhores.get(chave)
        if atual is None:
            melhores[chave] = item
            continue
        item_canonico = item['nome_pasta'] in PASTAS_CANONICAS
        atual_canonico = atual['nome_pasta'] in PASTAS_CANONICAS
        if item_canonico and not atual_canonico:
            melhores[chave] = item
        elif atual_canonico and not item_canonico:
            pass  # mantém o atual (é a pasta canônica)
        elif item['timestamp'] > atual['timestamp']:
            melhores[chave] = item

    descartados = [item for item in brutos if melhores[item['chave']] is not item]
    if descartados:
        print(f"\n⚠️  {len(descartados)} execução(ões) duplicada(s) descartada(s) "
              f"(mesmo algoritmo+redução+n_splits+test_size+dataset):")
        for item in descartados:
            vencedor = melhores[item['chave']]
            motivo = "é a execução já publicada no TCC" if vencedor['nome_pasta'] in PASTAS_CANONICAS else "execução mais antiga"
            print(f"   - {item['nome_pasta']} (timestamp {item['timestamp'] or '???'}) "
                  f"descartada em favor de {vencedor['nome_pasta']} "
                  f"(timestamp {vencedor['timestamp'] or '???'}, {motivo})")

    resultado = list(melhores.values())
    print(f"\n✅ {len(resultado)} experimento(s) único(s) carregado(s) de {len(brutos)} arquivo(s) válido(s) "
          f"({len(arquivos_json)} JSON encontrados em '{pasta_alvo}').")
    return resultado
