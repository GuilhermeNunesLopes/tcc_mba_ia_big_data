import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. Configurações de pastas
pasta_alvo = 'resultados'
pasta_saida = 'resultados/graficos_gerados'

# 🔴 Coloque aqui os nomes das pastas que você quer IGNORAR
pastas_ignoradas = ['resultados/resultados_antigos', 'resultados/historico_execucoes'] 

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

arquivos_json = []

# 2. Buscar arquivos ignorando as pastas indesejadas (inclui subpastas)
for root, dirs, files in os.walk(pasta_alvo):
    # Modifica a lista 'dirs' no próprio local para que o os.walk pule as pastas ignoradas
    dirs[:] = [d for d in dirs if d not in pastas_ignoradas]
    
    for file in files:
        if file.endswith('.json'):
            arquivos_json.append(os.path.join(root, file))

# Limita aos primeiros 4 arquivos encontrados (para o gráfico de 4 experimentos)
arquivos_json = arquivos_json[:4]

experimentos = []

if not arquivos_json:
    print(f"Nenhum arquivo JSON válido encontrado na pasta '{pasta_alvo}' (desconsiderando as ignoradas).")
else:
    print(f"Carregando {len(arquivos_json)} arquivos para comparação...\n")

    # 3. Ler os dados de cada arquivo
    for arquivo in arquivos_json:
        # Extrai o nome da pasta pai e o nome do arquivo para melhor identificação
        nome_pasta_pai = os.path.basename(os.path.dirname(arquivo))
        nome_arquivo = os.path.basename(arquivo)
        
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            if 'splits' in dados:
                df = pd.DataFrame(dados['splits'])
                config = dados.get('config', {})
                algoritmo = config.get('algoritmo', 'Alg').upper()
                reducao = config.get('reducao', 'Red').upper()
                
                # Cria um rótulo (label) para a legenda mostrando de qual pasta e arquivo veio
                label = f"{algoritmo}+{reducao} ({nome_pasta_pai}/{nome_arquivo.replace('.json', '')})"
                
                experimentos.append({'label': label, 'df': df})
                print(f" -> Carregado: {label}")
        except Exception as e:
            print(f" -> Erro ao ler {arquivo}: {e}")

    # 4. Gerar o Gráfico com 4 subplots (2x2)
    if experimentos:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Comparação de Modelos - Avaliação Walk-Forward', fontsize=18, fontweight='bold', y=0.98)
        
        cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # --- Subplot 1: PR_AUC ---
        for i, exp in enumerate(experimentos):
            axs[0, 0].plot(exp['df']['split'], exp['df']['PR_AUC'], marker='o', linewidth=2, color=cores[i], label=exp['label'])
        axs[0, 0].set_title('PR_AUC (Área sob a Curva Precision-Recall)', fontsize=13)
        axs[0, 0].set_ylabel('Valor', fontsize=11)
        axs[0, 0].set_ylim(-0.05, 1.05)
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)
        
        # --- Subplot 2: F1-Score ---
        for i, exp in enumerate(experimentos):
            axs[0, 1].plot(exp['df']['split'], exp['df']['F1_Score'], marker='D', linewidth=2, color=cores[i], label=exp['label'])
        axs[0, 1].set_title('F1-Score (Harmonia entre Precision e Recall)', fontsize=13)
        axs[0, 1].set_ylim(-0.05, 1.05)
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)
        
        # --- Subplot 3: Precision ---
        for i, exp in enumerate(experimentos):
            axs[1, 0].plot(exp['df']['split'], exp['df']['Precision'], marker='s', linewidth=2, color=cores[i], label=exp['label'])
        axs[1, 0].set_title('Precision (Precisão das Anomalias Encontradas)', fontsize=13)
        axs[1, 0].set_xlabel('Split (Avanço no Tempo)', fontsize=11)
        axs[1, 0].set_ylabel('Valor', fontsize=11)
        axs[1, 0].set_ylim(-0.05, 1.05)
        axs[1, 0].grid(True, linestyle='--', alpha=0.6)
        
        # --- Subplot 4: Recall ---
        for i, exp in enumerate(experimentos):
            axs[1, 1].plot(exp['df']['split'], exp['df']['Recall'], marker='^', linewidth=2, color=cores[i], label=exp['label'])
        axs[1, 1].set_title('Recall (Taxa de Anomalias Reais Detectadas)', fontsize=13)
        axs[1, 1].set_xlabel('Split (Avanço no Tempo)', fontsize=11)
        axs[1, 1].set_ylim(-0.05, 1.05)
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)

        # Legenda geral
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 5. Salvar o arquivo
        caminho_salvar = os.path.join(pasta_saida, 'comparacao_4_experimentos.png')
        plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nGráfico comparativo salvo com sucesso em: {caminho_salvar}")