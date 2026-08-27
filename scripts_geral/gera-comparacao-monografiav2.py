import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

experimentos = []

if not arquivos_json:
    print(f"Nenhum arquivo JSON válido encontrado na pasta '{pasta_alvo}'.")
else:
    print(f"Encontrados {len(arquivos_json)} arquivos. Carregando...\n")

    # 2. Ler os dados e armazenar as configurações
    for arquivo in arquivos_json:
        nome_arquivo = os.path.basename(arquivo)
        
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            if 'splits' in dados:
                df = pd.DataFrame(dados['splits'])
                
                # Extraindo todas as informações da configuração
                config = dados.get('config', {})
                algoritmo = config.get('algoritmo', 'Alg').upper()
                reducao = config.get('reducao', 'Sem_Reducao').upper()
                n_splits = config.get('n_splits', 'N/A')
                test_size = config.get('test_size', 'N/A')
                
                # NOVO FORMATO DA LEGENDA
                label = f"Alg: {algoritmo} | Red: {reducao} | Splits: {n_splits} | Test: {test_size}\n({nome_arquivo.replace('.json','')})"
                
                experimentos.append({
                    'label': label, 
                    'df': df,
                    'reducao': reducao
                })
        except Exception as e:
            print(f" -> Erro ao ler {arquivo}: {e}")

    # 3. Criar mapeamento dinâmico para as reduções
    reducoes_unicas = list(set([exp['reducao'] for exp in experimentos]))
    
    marcadores_disp = ['o', 's', '^', 'D', 'v', 'p', '*', 'X'] 
    linhas_disp = ['-', '--', '-.', ':'] 
    
    mapa_estilos = {}
    for i, red in enumerate(reducoes_unicas):
        mapa_estilos[red] = {
            'marker': marcadores_disp[i % len(marcadores_disp)],
            'linestyle': linhas_disp[i % len(linhas_disp)]
        }

    # Gerar Cores Distintas (Paleta Tab20)
    cmap = plt.get_cmap('tab20')
    cores = [cmap(i % 20) for i in range(len(experimentos))]

    # 4. Gerar o Gráfico
    if experimentos:
        fig, axs = plt.subplots(2, 2, figsize=(18, 12)) 
        fig.suptitle(f'Comparação Walk-Forward ({len(experimentos)} Experimentos)', fontsize=18, fontweight='bold', y=0.97)
        
        def plot_metrica(ax, coluna_metrica, titulo):
            for i, exp in enumerate(experimentos):
                red = exp['reducao']
                ax.plot(
                    exp['df']['split'], 
                    exp['df'][coluna_metrica], 
                    color=cores[i], 
                    marker=mapa_estilos[red]['marker'], 
                    linestyle=mapa_estilos[red]['linestyle'], 
                    linewidth=2.5, 
                    markersize=6,
                    label=exp['label'],
                    alpha=0.85
                )
            ax.set_title(titulo, fontsize=14)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xticks(experimentos[0]['df']['split'])
        
        # Plotando os subplots
        plot_metrica(axs[0, 0], 'PR_AUC', 'PR_AUC (Área sob a Curva Precision-Recall)')
        axs[0, 0].set_ylabel('Valor', fontsize=12)
        
        plot_metrica(axs[0, 1], 'F1_Score', 'F1-Score (Harmonia entre Precision e Recall)')
        
        plot_metrica(axs[1, 0], 'Precision', 'Precision (Precisão das Anomalias)')
        axs[1, 0].set_xlabel('Split (Avanço no Tempo)', fontsize=12)
        axs[1, 0].set_ylabel('Valor', fontsize=12)
        
        plot_metrica(axs[1, 1], 'Recall', 'Recall (Taxa de Anomalias Reais)')
        axs[1, 1].set_xlabel('Split (Avanço no Tempo)', fontsize=12)

        # 5. Legenda Inteligente e Ajuste de Layout
        handles, labels = axs[0, 0].get_legend_handles_labels()
        
        # Ajusta o número de colunas dependendo da quantidade de experimentos
        num_colunas = 2 if len(experimentos) <= 6 else 3
        if len(experimentos) > 12:
            num_colunas = 4
            
        fig.legend(
            handles, labels, 
            loc='upper center', 
            ncol=num_colunas, 
            fontsize=10, 
            bbox_to_anchor=(0.5, 0.05)
        )

        # Aumentei o espaço de fundo (0.12) para comportar a legenda que agora tem mais texto
        espaco_fundo = 0.12 + (len(experimentos) // num_colunas) * 0.03
        plt.tight_layout(rect=[0, espaco_fundo, 1, 0.95])
        
        caminho_salvar = os.path.join(pasta_saida, 'comparacao_multiplos_completa.png')
        plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nGráfico salvo em: {caminho_salvar}")