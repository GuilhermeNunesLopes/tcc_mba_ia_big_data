import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from _comparacao_utils import carregar_experimentos_walkforward

# 1. Configurações de pastas
# CORREÇÃO (27/08/2026): pasta_alvo era só 'resultados/comparar', o que
# excluía a rodada oficial PCA/5-splits/15% (guardada em 'resultados_antigos')
# — por isso o gráfico de 5 splits/teste=0.15 só mostrava barras de SVD, sem
# nenhuma barra de PCA para comparar. Agora varre 'resultados' inteiro.
pasta_alvo = 'resultados'
pasta_saida = 'resultados/graficos_gerados'
pastas_ignoradas = ['graficos_gerados']

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# 2. Carregar e deduplicar (ver _comparacao_utils.py) — corrige o bug em que
# duas pastas "walkforward_iforest_pca_n20_t4pct" podiam ser sobre datasets
# diferentes (BGL completo vs. BGL_filtrado) e ainda assim virar uma única
# barra "PCA", escondendo qual delas foi realmente usada.
experimentos_dedup = carregar_experimentos_walkforward(pasta_alvo, pastas_ignoradas)

metricas_ordem = ['Precision', 'Recall', 'F1_Score', 'PR_AUC']
nomes_metricas = ['Precision', 'Recall', 'F1-Score', 'PR-AUC']

dados_extraidos = []
for item in experimentos_dedup:
    resumo = item['dados']['resumo']
    medias = [resumo.get(m, {}).get('media', 0.0) for m in metricas_ordem]
    desvios = [resumo.get(m, {}).get('desvio_padrao', 0.0) for m in metricas_ordem]
    dados_extraidos.append({
        'algoritmo': item['algoritmo'],
        'reducao': item['reducao'],
        'n_splits': item['n_splits'],
        'test_size': item['test_size'],
        'linhas_dataset': item['linhas_dataset'],
        'medias': medias,
        'desvios': desvios,
    })

# 3. Agrupar por (n_splits, test_size, dataset) — o "dataset" (linhas brutas
# lidas) entra na chave do grupo justamente para NUNCA colocar, na mesma
# figura, uma barra tirada do BGL completo ao lado de uma barra tirada de
# uma amostra pequena só porque n_splits/test_size batem por coincidência.
grupos = defaultdict(list)
for item in dados_extraidos:
    chave_grupo = (item['n_splits'], item['test_size'], item['linhas_dataset'])
    grupos[chave_grupo].append(item)

cores_reducao = plt.get_cmap('tab10')

for (n_splits, test_size, linhas_dataset), experimentos_grupo in grupos.items():
    algoritmos_unicos = sorted(list(set([e['algoritmo'] for e in experimentos_grupo])))
    reducoes_unicas = sorted(list(set([e['reducao'] for e in experimentos_grupo])))

    n_algoritmos = len(algoritmos_unicos)
    if n_algoritmos == 0:
        continue

    if len(reducoes_unicas) < 2:
        print(f"\nℹ️  Grupo (n_splits={n_splits}, test_size={test_size}, dataset={linhas_dataset} linhas) "
              f"só tem {len(reducoes_unicas)} redutor ({', '.join(reducoes_unicas)}) — "
              f"gerado mesmo assim, mas não é uma comparação PCA vs. SVD completa "
              f"(falta rodar o(s) outro(s) redutor(es) sobre o MESMO dataset).")

    fig, axes = plt.subplots(1, n_algoritmos, figsize=(6 * n_algoritmos, 7), sharey=True)
    if n_algoritmos == 1:
        axes = [axes]

    titulo_principal = f"{' vs. '.join(reducoes_unicas)} — walk-forward ({n_splits} splits, teste={test_size}, dataset={linhas_dataset} linhas)"
    fig.suptitle(titulo_principal, fontsize=15, fontweight='bold', color='#1f4e79', y=0.98)
    fig.text(0.5, 0.93, "Métricas médias e desvio padrão extraídos da avaliação ao longo do tempo",
             ha='center', fontsize=11, fontstyle='italic', color='#555555')

    x_pos = np.arange(len(metricas_ordem))
    largura_barra = 0.8 / len(reducoes_unicas)

    mapa_cores = {red: cores_reducao(i) for i, red in enumerate(reducoes_unicas)}

    def escurecer_cor(rgba):
        return (rgba[0] * 0.7, rgba[1] * 0.7, rgba[2] * 0.7, 1.0)

    for i, alg in enumerate(algoritmos_unicos):
        ax = axes[i]
        dados_alg = [e for e in experimentos_grupo if e['algoritmo'] == alg]

        for j, red in enumerate(reducoes_unicas):
            exp_atual = next((e for e in dados_alg if e['reducao'] == red), None)
            if exp_atual:
                posicoes_barras = x_pos + (j - len(reducoes_unicas) / 2 + 0.5) * largura_barra
                cor_barra = mapa_cores[red]
                cor_erro = escurecer_cor(cor_barra)

                barras = ax.bar(posicoes_barras, exp_atual['medias'], largura_barra,
                                 yerr=exp_atual['desvios'], capsize=4,
                                 color=cor_barra, ecolor=cor_erro, error_kw={'elinewidth': 1.5},
                                 label=red if i == 0 else "")

                for k, barra in enumerate(barras):
                    y_max_erro = exp_atual['medias'][k] + exp_atual['desvios'][k]
                    valor_texto = f"{exp_atual['medias'][k]:.4f}"
                    if exp_atual['medias'][k] > 0:
                        ax.annotate(valor_texto,
                                    xy=(barra.get_x() + barra.get_width() / 2, y_max_erro),
                                    xytext=(0, 5), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9, color='#333333')

        ax.set_title(alg, fontsize=14, fontweight='bold', color='#1f4e79', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(nomes_metricas, fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#aaaaaa')
        ax.yaxis.grid(True, linestyle='-', color='#eeeeee', alpha=1.0)
        ax.set_axisbelow(True)

        if i == 0:
            ax.set_ylabel(f'Score (média ± desvio padrão, {n_splits} splits)', fontsize=11)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=len(reducoes_unicas),
               frameon=False, fontsize=11)

    plt.tight_layout(rect=[0, 0.0, 1, 0.85])

    nome_arq_saida = f"comparacao_barras_splits{n_splits}_teste{test_size}_linhas{linhas_dataset}.png"
    caminho_salvar = os.path.join(pasta_saida, nome_arq_saida)
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Gráfico de barras salvo: {caminho_salvar}")

print("\nProcessamento concluído!")
