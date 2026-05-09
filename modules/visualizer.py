import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_anomaly_distribution(df, output_path="distribuicao_scores.png"):
    """Mostra como o modelo separou o que é normal do que é anômalo."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='anomaly_score', hue='is_anomaly', element="step", palette="viridis")
    plt.title("Distribuição dos Scores de Anomalia")
    plt.xlabel("Score (Valores menores = Mais anômalos)")
    plt.ylabel("Frequência")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()

def plot_anomaly_timeline(df, output_path="linha_do_tempo.png"):
    """Plota os scores ao longo do tempo (índice). Útil para ver o efeito do Chaos Engineering."""
    plt.figure(figsize=(15, 6))
    
    # Plota a linha de score
    plt.plot(df.index, df['anomaly_score'], color='blue', alpha=0.5, label='Decision Score')
    
    # Destaca as anomalias em vermelho
    anomalias = df[df['is_anomaly'] == True]
    plt.scatter(anomalias.index, anomalias['anomaly_score'], color='red', label='Anomalia Detectada', s=20)
    
    plt.title("Linha do Tempo de Detecção de Anomalias")
    plt.xlabel("Sequência dos Logs")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_anomaly_counts(df, output_path="contagem_anomalias.png"):
    """Gráfico de barras para mostrar o desbalanceamento (Anomalias vs Normais)."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='is_anomaly', palette="magma")
    plt.title("Proporção: Logs Normais vs Anomalias")
    plt.xlabel("É Anomalia?")
    plt.ylabel("Quantidade de Logs")
    
    # Adiciona as porcentagens em cima das barras
    total = len(df)
    for p in plt.gca().patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        plt.gca().annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    plt.savefig(output_path)
    plt.close()