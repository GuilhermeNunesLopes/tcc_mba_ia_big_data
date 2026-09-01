import os
import pandas as pd
import matplotlib.pyplot as plt
from _comparacao_utils import carregar_experimentos_walkforward

# 1. Configurações de pastas
# CORREÇÃO (27/08/2026): pasta_alvo era só 'resultados/comparar', o que
# excluía execuções oficiais válidas guardadas em 'resultados/resultados_antigos'
# (ex.: a rodada PCA/5-splits/15% usada no TCC) e execuções soltas na raiz de
# 'resultados/' (ex.: o par SVD do BGL completo). Agora varre 'resultados'
# inteiro; a deduplicação em _comparacao_utils.py garante que duplicatas
# antigas/obsoletas continuem sendo descartadas, só que agora de forma
# explícita (avisando qual foi descartada) em vez de por exclusão de pasta.
pasta_alvo = 'resultados'
pasta_saida = 'resultados/graficos_gerados'
pastas_ignoradas = ['graficos_gerados']

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

experimentos_dedup = carregar_experimentos_walkforward(pasta_alvo, pastas_ignoradas)

experimentos = []
for item in experimentos_dedup:
    df = pd.DataFrame(item['dados']['splits'])
    label = (f"Alg: {item['algoritmo']} | Red: {item['reducao']} | "
             f"Splits: {item['n_splits']} | Test: {item['test_size']} | "
             f"Dataset: {item['linhas_dataset']} linhas\n({item['nome_pasta']})")
    experimentos.append({'label': label, 'df': df, 'reducao': item['reducao']})

if not experimentos:
    print(f"Nenhum experimento walk-forward válido encontrado na pasta '{pasta_alvo}'.")
else:
    # 2. Mapeamento dinâmico de estilo por redução (mesmo esquema do original)
    reducoes_unicas = sorted(list(set([exp['reducao'] for exp in experimentos])))

    marcadores_disp = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    linhas_disp = ['-', '--', '-.', ':']

    mapa_estilos = {}
    for i, red in enumerate(reducoes_unicas):
        mapa_estilos[red] = {
            'marker': marcadores_disp[i % len(marcadores_disp)],
            'linestyle': linhas_disp[i % len(linhas_disp)],
        }

    cmap = plt.get_cmap('tab20')
    cores = [cmap(i % 20) for i in range(len(experimentos))]

    # 3. Gerar o Gráfico
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Comparação Walk-Forward ({len(experimentos)} Experimentos únicos)',
                 fontsize=18, fontweight='bold', y=0.97)

    def plot_metrica(ax, coluna_metrica, titulo):
        for i, exp in enumerate(experimentos):
            red = exp['reducao']
            ax.plot(
                exp['df']['split'], exp['df'][coluna_metrica],
                color=cores[i], marker=mapa_estilos[red]['marker'],
                linestyle=mapa_estilos[red]['linestyle'],
                linewidth=2.5, markersize=6, label=exp['label'], alpha=0.85,
            )
        ax.set_title(titulo, fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(experimentos[0]['df']['split'])

    plot_metrica(axs[0, 0], 'PR_AUC', 'PR_AUC (Área sob a Curva Precision-Recall)')
    axs[0, 0].set_ylabel('Valor', fontsize=12)

    plot_metrica(axs[0, 1], 'F1_Score', 'F1-Score (Harmonia entre Precision e Recall)')

    plot_metrica(axs[1, 0], 'Precision', 'Precision (Precisão das Anomalias)')
    axs[1, 0].set_xlabel('Split (Avanço no Tempo)', fontsize=12)
    axs[1, 0].set_ylabel('Valor', fontsize=12)

    plot_metrica(axs[1, 1], 'Recall', 'Recall (Taxa de Anomalias Reais)')
    axs[1, 1].set_xlabel('Split (Avanço no Tempo)', fontsize=12)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    num_colunas = 2 if len(experimentos) <= 6 else 3
    if len(experimentos) > 12:
        num_colunas = 4

    fig.legend(handles, labels, loc='upper center', ncol=num_colunas, fontsize=9,
               bbox_to_anchor=(0.5, 0.04))

    espaco_fundo = 0.14 + (len(experimentos) // num_colunas) * 0.035
    plt.tight_layout(rect=[0, espaco_fundo, 1, 0.95])

    caminho_salvar = os.path.join(pasta_saida, 'comparacao_multiplos_completa.png')
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nGráfico salvo em: {caminho_salvar}")
