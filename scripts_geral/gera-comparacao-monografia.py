import os
import pandas as pd
import matplotlib.pyplot as plt
from _comparacao_utils import carregar_experimentos_walkforward

# 1. Configurações de pastas
pasta_alvo = 'resultados'
pasta_saida = 'resultados/graficos_gerados'

# 🔴 Coloque aqui os nomes das pastas que você quer IGNORAR
# (CORREÇÃO 27/08/2026: 'resultados_antigos' deixou de ser ignorada — ela tem
# execuções oficiais válidas, ex.: a rodada PCA/5-splits/15% usada no TCC. A
# deduplicação de carregar_experimentos_walkforward() já cuida de descartar
# duplicatas antigas/obsoletas dentro dela pela combinação exata de
# algoritmo+redução+n_splits+test_size+dataset, mantendo só a mais recente.)
pastas_ignoradas = ['graficos_gerados']

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# 2. Carregar e deduplicar (ver _comparacao_utils.py) — corrige o bug em que
# duas pastas com nome parecido podiam ser sobre datasets diferentes e ainda
# assim serem comparadas como se fossem a mesma coisa.
experimentos_dedup = carregar_experimentos_walkforward(pasta_alvo, pastas_ignoradas)

# 3. Monta os objetos de plotagem (label + DataFrame de splits) a partir dos
# experimentos deduplicados, ordenados do mais recente para o mais antigo —
# mantém o espírito original do script ("os últimos experimentos rodados"),
# só que agora sem risco de pegar duas execuções conflitantes com o mesmo
# nome de pasta.
experimentos_dedup.sort(key=lambda item: item['timestamp'], reverse=True)

MAX_EXPERIMENTOS = 4
if len(experimentos_dedup) > MAX_EXPERIMENTOS:
    print(f"\nℹ️  {len(experimentos_dedup)} experimentos únicos encontrados — "
          f"mostrando os {MAX_EXPERIMENTOS} mais recentes (gráfico de 4 experimentos). "
          f"Os demais não entraram neste PNG (use gera-comparacao-monografiav2.py "
          f"para ver todos de uma vez).")
experimentos_escolhidos = experimentos_dedup[:MAX_EXPERIMENTOS]

experimentos = []
for item in experimentos_escolhidos:
    df = pd.DataFrame(item['dados']['splits'])
    label = (f"{item['algoritmo']}+{item['reducao']} "
             f"({item['nome_pasta']}, dataset={item['linhas_dataset']} linhas)")
    experimentos.append({'label': label, 'df': df})
    print(f" -> Incluído: {label}")

if not experimentos:
    print(f"Nenhum experimento walk-forward válido encontrado na pasta '{pasta_alvo}'.")
else:
    # 4. Gerar o Gráfico (mesmo layout 2x2 do script original)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Comparação de Modelos - Avaliação Walk-Forward', fontsize=16, fontweight='bold')

    cores = plt.get_cmap('tab10')
    marcadores = ['o', 's', '^', 'D']

    def plot_metrica(ax, coluna_metrica, titulo):
        for i, exp in enumerate(experimentos):
            ax.plot(
                exp['df']['split'], exp['df'][coluna_metrica],
                marker=marcadores[i % len(marcadores)], color=cores(i % 10),
                linewidth=2, markersize=6, label=exp['label'], alpha=0.85,
            )
        ax.set_title(titulo, fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)

    plot_metrica(axs[0, 0], 'PR_AUC', 'PR_AUC (Área sob a Curva Precision-Recall)')
    axs[0, 0].set_ylabel('Valor')

    plot_metrica(axs[0, 1], 'F1_Score', 'F1-Score (Harmonia entre Precision e Recall)')

    plot_metrica(axs[1, 0], 'Precision', 'Precision (Precisão das Anomalias Encontradas)')
    axs[1, 0].set_xlabel('Split (Avanço no Tempo)')
    axs[1, 0].set_ylabel('Valor')

    plot_metrica(axs[1, 1], 'Recall', 'Recall (Taxa de Anomalias Reais Detectadas)')
    axs[1, 1].set_xlabel('Split (Avanço no Tempo)')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    caminho_salvar = os.path.join(pasta_saida, 'comparacao_4_experimentos.png')
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nGráfico comparativo salvo com sucesso em: {caminho_salvar}")
