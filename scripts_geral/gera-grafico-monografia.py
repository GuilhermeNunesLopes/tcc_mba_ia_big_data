import matplotlib.pyplot as plt
import numpy as np

# Dados da sua tabela
labels = [
    'IForest, SVD 39 comp.', 'OCSVM, treino 20k (graficos)', 
    'Percentil 50%', 'Percentil 60%', 
    'Literatura (100% base)', 'OCSVM, amostra 20k (svm)', 
    'OCSVM, amostra 33k', 'OCSVM, amostra 40k'
]

precision = [0.9637, 0.8199, 0.9594, 0.8239, 0.8774, 0.8129, 0.8104, 0.8081]
recall = [0.9723, 0.8419, 0.9642, 1.0000, 0.4947, 0.8419, 0.8354, 0.8104]
f1 = [0.9680, 0.8308, 0.9618, 0.9034, 0.6327, 0.8272, 0.8227, 0.8092]

x = np.arange(len(labels))  # Localização das labels no eixo x
width = 0.25  # Largura das barras

# Criando a figura e os eixos
fig, ax = plt.subplots(figsize=(12, 6))

# Adicionando as barras
rects1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
rects2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ca02c')

# Customizando o gráfico
ax.set_ylabel('Pontuação')
ax.set_title('Comparativo de Desempenho por Configuração')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.1) # Vai até 1.1 para dar espaço para a legenda

# Linhas de grade para facilitar a leitura
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Ajustando o layout para não cortar os textos e salvando
fig.tight_layout()
plt.savefig('grafico_resultados.png', dpi=300)
plt.show()