import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. Configurações de pastas
pasta_alvo = 'resultados/comparar'
pasta_saida = 'resultados/graficos_gerados'

# 🔴 Coloque aqui os nomes das pastas que você quer IGNORAR
pastas_ignoradas = ['resultados/resultados_antigos', 'resultados/historico_execucoes'] 

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

arquivos_json = []

for root, dirs, files in os.walk(pasta_alvo):
    dirs[:] = [d for d in dirs if d not in pastas_ignoradas]
    for file in files:
        if file.endswith('.json'):
            arquivos_json.append(os.path.join(root, file))

# Lista das métricas na ordem exata da imagem de referência
metricas_ordem = ['Precision', 'Recall', 'F1_Score', 'PR_AUC']
nomes_metricas = ['Precision', 'Recall', 'F1-Score', 'PR-AUC']

# 2. Ler os arquivos e organizar os dados
dados_extraidos = []

for arquivo in arquivos_json:
    try:
        with open(arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        if 'resumo' in dados and 'config' in dados:
            config = dados['config']
            resumo = dados['resumo']
            
            algoritmo = config.get('algoritmo', 'Alg Desconhecido').upper()
            reducao = config.get('reducao', 'Sem_Reducao').upper()
            n_splits = config.get('n_splits', 'N/A')
            test_size = config.get('test_size', 'N/A')
            
            # Extrair médias e desvios
            medias = []
            desvios = []
            for metrica in metricas_ordem:
                medias.append(resumo.get(metrica, {}).get('media', 0.0))
                desvios.append(resumo.get(metrica, {}).get('desvio_padrao', 0.0))
                
            dados_extraidos.append({
                'algoritmo': algoritmo,
                'reducao': reducao,
                'n_splits': n_splits,
                'test_size': test_size,
                'medias': medias,
                'desvios': desvios
            })
    except Exception as e:
        print(f"Erro ao ler {arquivo}: {e}")

# 3. Agrupar por (n_splits, test_size)
grupos = defaultdict(list)
for item in dados_extraidos:
    chave_grupo = (item['n_splits'], item['test_size'])
    grupos[chave_grupo].append(item)

# 4. Gerar os gráficos para cada grupo
cores_reducao = plt.get_cmap('tab10') # Paleta de cores para as barras

for (n_splits, test_size), experimentos_grupo in grupos.items():
    # Encontrar todos os algoritmos e reduções únicos neste grupo
    algoritmos_unicos = sorted(list(set([e['algoritmo'] for e in experimentos_grupo])))
    reducoes_unicas = sorted(list(set([e['reducao'] for e in experimentos_grupo])))
    
    n_algoritmos = len(algoritmos_unicos)
    
    if n_algoritmos == 0:
        continue

    # Criar a figura (Side-by-side dependendo do número de algoritmos)
    fig, axes = plt.subplots(1, n_algoritmos, figsize=(6 * n_algoritmos, 7), sharey=True)
    
    # Se houver apenas 1 algoritmo, o axes não será uma lista, então forçamos para ser
    if n_algoritmos == 1:
        axes = [axes]

    # Título Principal (estilo relatório)
    titulo_principal = f"{' vs. '.join(reducoes_unicas)} — walk-forward ({n_splits} splits, teste={test_size})"
    fig.suptitle(titulo_principal, fontsize=16, fontweight='bold', color='#1f4e79', y=0.98)
    
    # Subtítulo opcional (pode ser ajustado conforme a necessidade)
    fig.text(0.5, 0.93, "Métricas médias e desvio padrão extraídos da avaliação ao longo do tempo", 
             ha='center', fontsize=11, fontstyle='italic', color='#555555')

    # Configurações do layout de barras
    x_pos = np.arange(len(metricas_ordem))  # Posições das métricas no eixo X
    largura_barra = 0.8 / len(reducoes_unicas) # Ajusta a largura dependendo de quantas reduções existirem

    # Mapear cores para as reduções
    mapa_cores = {red: cores_reducao(i) for i, red in enumerate(reducoes_unicas)}
    # Mapear cor do erro (uma versão um pouco mais escura da cor da barra para dar destaque)
    def escurecer_cor(rgba):
        return (rgba[0]*0.7, rgba[1]*0.7, rgba[2]*0.7, 1.0)

    # Plotar para cada algoritmo (Subplot)
    for i, alg in enumerate(algoritmos_unicos):
        ax = axes[i]
        
        # Filtrar dados para este algoritmo
        dados_alg = [e for e in experimentos_grupo if e['algoritmo'] == alg]
        
        for j, red in enumerate(reducoes_unicas):
            # Encontrar o experimento específico (algoritmo + redução)
            exp_atual = next((e for e in dados_alg if e['reducao'] == red), None)
            
            if exp_atual:
                posicoes_barras = x_pos + (j - len(reducoes_unicas)/2 + 0.5) * largura_barra
                
                cor_barra = mapa_cores[red]
                cor_erro = escurecer_cor(cor_barra)
                
                # Plotando a barra com a barra de erro (yerr)
                barras = ax.bar(posicoes_barras, exp_atual['medias'], largura_barra, 
                                yerr=exp_atual['desvios'], capsize=4, 
                                color=cor_barra, ecolor=cor_erro, error_kw={'elinewidth': 1.5},
                                label=red if i == 0 else "") # Legenda apenas no 1º subplot
                
                # Adicionando o texto (rótulo de valor) acima do desvio padrão
                for k, barra in enumerate(barras):
                    y_max_erro = exp_atual['medias'][k] + exp_atual['desvios'][k]
                    valor_texto = f"{exp_atual['medias'][k]:.4f}"
                    # Para não cruzar com outras informações, se o valor for 0, oculta ou bota no zero
                    if exp_atual['medias'][k] > 0:
                        ax.annotate(valor_texto,
                                    xy=(barra.get_x() + barra.get_width() / 2, y_max_erro),
                                    xytext=(0, 5), # 5 pontos acima do topo da barra de erro
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9, color='#333333')

        # Estilo do Subplot (similar ao da imagem de referência)
        ax.set_title(alg, fontsize=14, fontweight='bold', color='#1f4e79', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(nomes_metricas, fontsize=11)
        
        # Limpar bordas superior e direita
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#aaaaaa')
        
        # Grid horizontal sutil
        ax.yaxis.grid(True, linestyle='-', color='#eeeeee', alpha=1.0)
        ax.set_axisbelow(True) # Faz o grid ficar atrás das barras
        
        # Adicionar o rótulo do eixo Y apenas no primeiro gráfico
        if i == 0:
            ax.set_ylabel(f'Score (média ± desvio padrão, {n_splits} splits)', fontsize=11)

    # Adicionar a legenda global no topo
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=len(reducoes_unicas), 
               frameon=False, fontsize=11)

    # Ajustar layout
    plt.tight_layout(rect=[0, 0.0, 1, 0.85]) # Deixa espaço no topo para título e legenda
    
    # Salvar
    nome_arq_saida = f"comparacao_barras_splits{n_splits}_teste{test_size}.png"
    caminho_salvar = os.path.join(pasta_saida, nome_arq_saida)
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Gráfico de barras salvo: {caminho_salvar}")

print("\nProcessamento concluído!")