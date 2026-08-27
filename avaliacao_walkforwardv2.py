"""
Avaliação Walk-Forward (múltiplos splits cronológicos).

Em vez de confiar num único corte 80/20, este script treina e avalia o
pipeline várias vezes ao longo do tempo (janela de treino expansiva,
janela de teste deslizando para frente), e reporta média ± desvio padrão
das métricas entre os splits. Isso responde à pergunta natural de banca:
"e se você tivesse cortado os dados em outro ponto, o resultado seria
parecido?".

Uso:
    python avaliacao_walkforward.py --n-splits 5 --test-size 0.15 --algoritmo iforest --reducao pca

Requer um dataset com coluna de rótulo (Label/Anomaly/...) para que as
métricas (F1/Precision/Recall/PR-AUC) façam sentido — ex.: o BGL baixado
via dowloand_dataset_hugging.py. Sem rótulo, o script ainda roda (o
detector continua funcionando), mas não há como calcular essas métricas.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import modules.parse_system as parse_system
import modules.visualizer as visualizer
import modules.run_output as run_output
import pipeline as pipeline


# Arquivos de log lidos por esta avaliação — em vez de varrer uma pasta
# inteira (que também pegaria os .csv de structured/templates que ficam
# junto dos .log em logpai\BGL), a lista é explícita e fixa.
ARQUIVOS_LOG_PADRAO = [
    os.path.join("logpai", "BGL", "BGL200k.log"),
    os.path.join("logpai", "BGL", "BGL500k.log"),
    os.path.join("logpai", "BGL", "BGL900k.log"),
]


def ler_configuracoes_padrao():
    """Lê algoritmo/redução/contaminação do config.json, se existir (mesmo arquivo do dashboard)."""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            return (
                config.get("taxa_contaminacao", "auto"),
                config.get("algoritmo", "iforest"),
                config.get("reducao", "pca"),
            )
    except (FileNotFoundError, json.JSONDecodeError):
        return "auto", "iforest", "pca"


def coletar_logs(arquivos_ativos, resumos_parse_saida=None):
    """
    Lê a lista de arquivos de log informada e retorna o DataFrame concatenado
    de logs parseados pelo Drain3.

    resumos_parse_saida: lista opcional, passada pelo chamador. Recebe um
    dict de resumo (linhas brutas -> templates únicos, ver
    modules/parse_system.py) por arquivo parseado, para o chamador salvar
    isso junto do resultado da execução.
    """
    df_list = []
    for caminho_arquivo in arquivos_ativos:
        if not os.path.isfile(caminho_arquivo):
            print(f"⚠️  Arquivo não encontrado, pulando: {caminho_arquivo}")
            continue

        # ACHADO CORRIGIDO: nome de fonte isolado do motor ao vivo
        # (main_v4/v5/v6), que usa o mesmo nome de arquivo como nome_fonte e
        # mantém um estado do Drain3 persistido e "aquecido" com o histórico
        # inteiro do arquivo. Se a avaliação walk-forward reusasse esse mesmo
        # estado (drain3_state_<arquivo>.bin), os splits mais antigos se
        # beneficiariam de templates que só existem porque o motor ao vivo já
        # tinha processado o arquivo inteiro antes — inclusive trechos que,
        # para aquele split, representam "o futuro". Um nome de fonte próprio
        # grava num arquivo .bin separado, nunca tocado pelo motor ao vivo.
        nome_da_fonte = os.path.splitext(os.path.basename(caminho_arquivo))[0] + "_walkforward_eval"

        # Além de isolar do motor ao vivo, apaga o estado de uma execução
        # anterior desta MESMA avaliação antes de começar: sem isso, uma
        # segunda rodada do walk-forward herdaria os templates aprendidos na
        # primeira (que já tinha visto o arquivo até o fim) — mesmo tipo de
        # vazamento, só que entre execuções do próprio script. Cada execução
        # aprende os templates do zero, só a partir da leitura sequencial do
        # arquivo (que já respeita a ordem cronológica) — ou seja, o template
        # usado numa linha nunca depende de linhas futuras dentro desta mesma
        # execução.
        caminho_estado = os.path.join(
            parse_system.DIRETORIO_ESTADOS_PADRAO, f"drain3_state_{nome_da_fonte}.bin"
        )
        if os.path.isfile(caminho_estado):
            os.remove(caminho_estado)

        resumo_arquivo = {} if resumos_parse_saida is not None else None
        gerador_lotes = parse_system.automatic_drain_parse(
            caminho_arquivo, nome_fonte=nome_da_fonte, tamanho_lote=100000,
            resumo_saida=resumo_arquivo,
        )
        for df_lote in gerador_lotes:
            if not df_lote.empty:
                df_lote['Source_Folder'] = caminho_arquivo
                df_list.append(df_lote)
        if resumos_parse_saida is not None and resumo_arquivo:
            resumos_parse_saida.append(resumo_arquivo)

    if not df_list:
        return pd.DataFrame()
    df_logs = pd.concat(df_list, ignore_index=True)
    return extrair_rotulo_bgl(df_logs)


def extrair_rotulo_bgl(df_logs):
    """
    O BGL (LogHub) grava o rótulo como o PRIMEIRO token de cada linha bruta:
    "-" para normal, ou um código de alerta (ex.: "APPREAD", "KERNDTLB") para
    anomalia real, seguido do timestamp em epoch — ex.:
        - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ...
        APPREAD 1117869872 2005.06.04 R04-M1-N4-I:J18-U11 ...
    Nem automatic_drain_parse() nem preprocessar_logs_brutos() extraem esse
    rótulo (o pipeline genérico não é específico de um dataset) — sem essa
    coluna 'Label', extrair_rotulo() em pipeline.py nunca encontra rótulo,
    tem_rotulo fica False, e os gráficos da seção 3.7 nunca são gerados.
    Só ativa a extração se o padrão bater em quase todas as linhas do lote
    (>=90%), para não injetar uma coluna 'Label' incorreta em datasets que
    não seguem o formato do BGL (nesse caso, o pipeline segue como antes,
    em modo puramente não supervisionado).
    """
    if df_logs.empty or 'Raw_Log' not in df_logs.columns:
        return df_logs

    tokens_label = df_logs['Raw_Log'].str.extract(r'^(\S+)\s+\d{10}\s+\d{4}\.\d{2}\.\d{2}\s')[0]
    taxa_casamento = tokens_label.notna().mean()

    if taxa_casamento >= 0.90:
        print(f"🏷️  Rótulo BGL detectado no formato bruto ({taxa_casamento:.1%} das linhas casam o padrão) "
              f"— coluna 'Label' adicionada para a avaliação walk-forward.")
        df_logs = df_logs.copy()
        df_logs['Label'] = (tokens_label != '-').astype(int)

    return df_logs


def gerar_splits_walkforward(df_logs, n_splits=5, test_size=0.15):
    """
    Gera múltiplos splits cronológicos por walk-forward (janela de treino
    expansiva). Cada split usa uma janela de teste de tamanho fixo,
    deslizando para frente no tempo; o treino de cada split usa TUDO que
    veio antes daquela janela (nunca dados do futuro).

    df_logs precisa já estar ordenado cronologicamente (preprocessar_logs_brutos faz isso).
    """
    n = len(df_logs)
    tamanho_teste = int(n * test_size)
    if tamanho_teste < 1:
        raise ValueError("Dataset pequeno demais para o test_size informado.")

    inicio_minimo_treino = n - n_splits * tamanho_teste
    if inicio_minimo_treino < tamanho_teste:
        raise ValueError(
            f"Dataset com {n} linhas é pequeno demais para {n_splits} splits "
            f"com test_size={test_size}. Reduza --n-splits ou --test-size."
        )

    splits = []
    for k in range(n_splits):
        fim_treino = inicio_minimo_treino + k * tamanho_teste
        inicio_teste = fim_treino
        fim_teste = inicio_teste + tamanho_teste

        df_train = df_logs.iloc[:fim_treino].reset_index(drop=True)
        df_test = df_logs.iloc[inicio_teste:fim_teste].reset_index(drop=True)
        splits.append((df_train, df_test))

    return splits


def rodar_avaliacao_walkforward(arquivos, algoritmo="iforest", reducao="pca",
                                 taxa_contaminacao="auto", n_splits=5, test_size=0.15,
                                 caminho_saida="resultados/avaliacao_walkforward.json"):
    print(f"📥 Coletando e parseando logs de {len(arquivos)} arquivo(s)...")
    resumos_parse = []
    df_logs = coletar_logs(arquivos, resumos_parse_saida=resumos_parse)
    if df_logs.empty:
        raise RuntimeError("Nenhum log encontrado nos arquivos configurados.")

    print("🛡️ Aplicando Whitelist e Engenharia de Features Temporais...")
    df_logs = pipeline.preprocessar_logs_brutos(df_logs)
    print(f"Total de logs após pré-processamento: {len(df_logs)}")

    splits = gerar_splits_walkforward(df_logs, n_splits=n_splits, test_size=test_size)

    resultados_por_split = []
    tem_rotulo = False
    # Acumula evidência visual ao longo dos splits: matriz de confusão somada
    # (todos os splits) e o df_resultado do último split (para o gráfico de
    # comparativo real-vs-detectado). Nada disso influencia treino/threshold —
    # é só para os gráficos gerados no fim.
    cm_total = np.zeros((2, 2), dtype=int)
    df_resultado_ultimo_split = None
    # Acumula y_true/score de TODOS os splits com rótulo (held-out real,
    # nunca visto no treino daquele split) para a curva Precision-Recall
    # combinada gerada no fim — concatenação dos mesmos arrays que cada
    # split já calcula dentro de pipeline.treinar_e_avaliar, não um cálculo
    # à parte.
    y_true_acumulado = []
    scores_acumulado = []

    for i, (df_train, df_test) in enumerate(splits, start=1):
        print(f"\n===== SPLIT {i}/{n_splits} =====")
        print(f"Treino: {len(df_train)} logs | Teste: {len(df_test)} logs")

        resultado = pipeline.treinar_e_avaliar(
            df_train, df_test,
            taxa_contaminacao_ativa=taxa_contaminacao,
            algoritmo_ativo=algoritmo,
            reducao_ativa=reducao,
            calcular_rca=False,  # não precisa de DBSCAN/RCA para avaliação de métricas
        )

        metricas = dict(resultado["metricas_ml"] or {})
        df_resultado = resultado["df_resultado"]

        if resultado["y_verdadeiro"] is not None:
            tem_rotulo = True
            if {"y_true_label", "pred_is_anomaly"}.issubset(df_resultado.columns):
                cm_total += confusion_matrix(
                    df_resultado["y_true_label"], df_resultado["pred_is_anomaly"], labels=[0, 1]
                )
            if {"y_true_label", "anomaly_score"}.issubset(df_resultado.columns):
                # -anomaly_score: mesma convenção usada em todo o pipeline
                # (quanto maior, mais anômalo) — ver anomaly_detector.py,
                # scores_para_curva.
                y_true_acumulado.append(df_resultado["y_true_label"].values)
                scores_acumulado.append(-df_resultado["anomaly_score"].values)
            df_resultado_ultimo_split = df_resultado

        metricas["split"] = i
        metricas["n_treino"] = len(df_train)
        metricas["n_teste"] = len(df_test)
        resultados_por_split.append(metricas)

        if metricas.get("F1_Score") is not None:
            print(f"  F1={metricas['F1_Score']:.4f} | Precision={metricas['Precision']:.4f} | "
                  f"Recall={metricas['Recall']:.4f} | PR_AUC={metricas['PR_AUC']:.4f}")
        else:
            print("  (sem rótulo neste split — só detecção rodou, sem métrica supervisionada)")

    if not tem_rotulo:
        print("\n⚠️  Nenhum split encontrou coluna de rótulo (Label/Anomaly/...). "
              "A avaliação walk-forward roda, mas não há F1/Precision/Recall para agregar. "
              "Use um dataset rotulado (ex.: BGL) para essa análise.")

    df_resultados = pd.DataFrame(resultados_por_split)
    resumo = {}
    for col in ["F1_Score", "Precision", "Recall", "PR_AUC"]:
        if col in df_resultados.columns and df_resultados[col].notna().any():
            resumo[col] = {
                "media": float(df_resultados[col].mean()),
                "desvio_padrao": float(df_resultados[col].std(ddof=0)),
                "min": float(df_resultados[col].min()),
                "max": float(df_resultados[col].max()),
            }

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    saida = {
        "splits": resultados_por_split,
        "resumo": resumo,
        "config": {
            "algoritmo": algoritmo, "reducao": reducao,
            "n_splits": n_splits, "test_size": test_size,
            "taxa_contaminacao": taxa_contaminacao,
        },
        # Resumo do parse Drain3 (linhas brutas -> templates únicos) por
        # arquivo de origem, salvo junto do resultado — ver modules/parse_system.py.
        "resumo_parse_drain3": resumos_parse,
    }
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)

    print(f"\n===== RESUMO — {n_splits} splits (média ± desvio padrão) =====")
    if resumo:
        for metrica, valores in resumo.items():
            print(f"{metrica}: {valores['media']:.4f} ± {valores['desvio_padrao']:.4f}  "
                  f"(min={valores['min']:.4f}, max={valores['max']:.4f})")
    print(f"\nResultado completo salvo em: {caminho_saida}")

    # ===================================================================
    # GRÁFICOS (evidência visual para a monografia/banca)
    # ===================================================================
    if tem_rotulo:
        print("\n📈 Gerando gráficos da avaliação walk-forward...")
        pasta_graficos = os.path.dirname(caminho_saida) or "resultados"
        os.makedirs(pasta_graficos, exist_ok=True)

        fig_splits = visualizer.plot_walkforward_metricas_por_split(df_resultados)
        if fig_splits is not None:
            caminho_fig_splits = os.path.join(pasta_graficos, "grafico_walkforward_metricas_por_split.html")
            fig_splits.write_html(caminho_fig_splits)
            print(f"   -> {caminho_fig_splits} (F1/Precision/Recall/PR-AUC por split, com médias)")

        if cm_total.sum() > 0:
            caminho_fig_cm = os.path.join(pasta_graficos, "grafico_walkforward_matriz_confusao.html")
            visualizer.plot_confusion_matrix_plotly(cm_total).write_html(caminho_fig_cm)
            print(f"   -> {caminho_fig_cm} (soma da matriz de confusão de todos os splits)")

        if df_resultado_ultimo_split is not None:
            caminho_fig_comp = os.path.join(pasta_graficos, "grafico_walkforward_comparativo_ultimo_split.html")
            visualizer.plot_comparativo_antes_depois(df_resultado_ultimo_split).write_html(caminho_fig_comp)
            print(f"   -> {caminho_fig_comp} (real vs. detectado, último split)")

        if y_true_acumulado:
            # threshold_usado fica de fora aqui de propósito: cada split treina
            # um Isolation Forest independente, com sua própria escala de
            # decision_function — o threshold de UM split não tem leitura
            # válida no eixo de score combinado dos outros splits. Para
            # literatura_v2.py (um único modelo) essa marcação faz sentido;
            # aqui, só a curva e o "melhor F1 possível" são comparáveis.
            caminho_fig_pr = os.path.join(pasta_graficos, "pr_curve_walkforward_bgl.png")
            visualizer.plot_pr_curve_threshold(
                y_true=np.concatenate(y_true_acumulado),
                scores=np.concatenate(scores_acumulado),
                titulo=f"Walk-Forward — held-out combinado dos {n_splits} splits ({algoritmo}/{reducao})",
                caminho_saida=caminho_fig_pr,
            )
    else:
        print("\n⏸️ Sem rótulo em nenhum split — pulando geração de gráficos (precisam de Ground Truth).")

    return saida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação walk-forward do detector de anomalias em logs.")
    parser.add_argument("--arquivos", type=str, default=None,
                         help="Caminhos de arquivo separados por vírgula. Se omitido, usa ARQUIVOS_LOG_PADRAO.")
    parser.add_argument("--n-splits", type=int, default=5, help="Número de splits cronológicos (padrão: 5).")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fração de cada janela de teste (padrão: 0.15).")
    parser.add_argument("--algoritmo", type=str, default=None, choices=["iforest", "ocsvm"])
    parser.add_argument("--reducao", type=str, default=None, choices=["pca", "svd"])
    parser.add_argument("--contaminacao", type=str, default=None,
                         help="Taxa de contaminação (ex.: 0.03) ou 'auto'.")

    args = parser.parse_args()

    contaminacao_config, algoritmo_config, reducao_config = ler_configuracoes_padrao()

    arquivos = args.arquivos.split(",") if args.arquivos else ARQUIVOS_LOG_PADRAO
    algoritmo = args.algoritmo or algoritmo_config
    reducao = args.reducao or reducao_config
    if args.contaminacao is not None:
        try:
            contaminacao = float(args.contaminacao)
        except ValueError:
            contaminacao = args.contaminacao
    else:
        contaminacao = contaminacao_config

    if not arquivos:
        raise SystemExit("Nenhum arquivo configurado. Use --arquivos ou edite ARQUIVOS_LOG_PADRAO.")

    # Pasta nova por execução (data/hora + parâmetros no nome) — permite
    # comparar execuções diferentes (outro n_splits/test_size/algoritmo) sem
    # uma sobrescrever a saída da outra.
    tag = f"walkforward_{algoritmo}_{reducao}_n{args.n_splits}_t{int(args.test_size * 100)}pct"
    pasta_execucao = run_output.criar_pasta_execucao(tag)

    rodar_avaliacao_walkforward(
        arquivos=arquivos,
        algoritmo=algoritmo,
        reducao=reducao,
        taxa_contaminacao=contaminacao,
        n_splits=args.n_splits,
        test_size=args.test_size,
        caminho_saida=os.path.join(pasta_execucao, "avaliacao_walkforward.json"),
    )