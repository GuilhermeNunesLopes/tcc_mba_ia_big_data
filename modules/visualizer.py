import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import plotly.figure_factory as ff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import os
import json

def plot_anomaly_timeline_plotly(df):
    """Gera uma linha do tempo otimizada com cores de alto contraste."""
    if df.empty:
        return go.Figure()
        
    df_plot = df.copy()
    
    # Rótulo em texto legível para o Plotly separar as séries sem ambiguidade
    df_plot['Status'] = df_plot['pred_is_anomaly'].apply(
        lambda x: 'Anomalia' if str(x) in ['1', 'True', 'true'] else 'Normal'
    )
    df_plot['tamanho_ponto'] = df_plot['Status'].apply(lambda x: 12 if x == 'Anomalia' else 5)

    # Garante que as anomalias sejam desenhadas POR CIMA dos pontos normais
    normais = df_plot[df_plot['Status'] == 'Normal']
    anomalias = df_plot[df_plot['Status'] == 'Anomalia']

    if len(normais) > 5000:
        normais = normais.sample(n=5000, random_state=42)
        
    df_final_plot = pd.concat([normais, anomalias])
    
    tem_timestamp = 'Timestamp' in df_final_plot.columns
    x_col = 'Timestamp' if tem_timestamp else df_final_plot.index
    x_label = 'Tempo (Hora do Log)' if tem_timestamp else 'Sequência dos Logs'
    
    hover_cols = []
    if 'Template' in df_final_plot.columns: hover_cols.append('Template')
    if 'Source_Folder' in df_final_plot.columns: hover_cols.append('Source_Folder')

    fig = px.scatter(
        df_final_plot, 
        x=x_col, 
        y='anomaly_score', 
        color='Status',
        color_discrete_map={'Normal': '#00FF00', 'Anomalia': '#FF0000'}, 
        size='tamanho_ponto',
        title="Linha do Tempo de Detecção de Anomalias",
        labels={x_col: x_label, 'anomaly_score': 'Decision Score (Gravidade)'},
        hover_data=hover_cols
    )
    
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )
    
    if tem_timestamp:
        fig.update_xaxes(rangeslider_visible=True)
        
    return fig

def plot_metricas_historico(historico_path="resultados/historico_metricas.json"):
    """Gera um gráfico de linha mostrando a evolução das métricas ao longo dos lotes."""
    if not os.path.exists(historico_path):
        return None
        
    try:
        with open(historico_path, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            
        df_hist = pd.DataFrame(dados)
        
        if df_hist.empty or 'Silhouette_Score' not in df_hist.columns:
            return None
            
        # Remove os lotes onde não houve clusterização (NaN)
        df_hist = df_hist.dropna(subset=['Silhouette_Score'])
        
        if df_hist.empty:
            return None
            
        fig = px.line(
            df_hist, 
            x='Timestamp_Lote', 
            y='Silhouette_Score',
            markers=True,
            title="Evolução do Silhouette Score (Qualidade da Clusterização)",
            labels={'Timestamp_Lote': 'Horário do Lote', 'Silhouette_Score': 'Silhouette Score (Coesão)'}
        )
        
        # Linha azul neon com marcadores vermelhos
        fig.update_traces(line_color='#00F2FE', marker=dict(size=10, color='#FF4B4B'))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            yaxis=dict(range=[-1, 1]) # O Score sempre varia de -1 a 1
        )
        return fig
    except Exception as e:
        print(f"Erro ao plotar histórico: {e}")
        return None

def plot_anomaly_distribution_plotly(df):
    """Gera um histograma interativo e super otimizado para a Web."""
    # 1. DOWNSAMPLING PARA O HISTOGRAMA
    #anomalias = df[df['pred_is_anomaly'] == True]
    #normais = df[df['pred_is_anomaly'] == False]
    anomalias = df[df['pred_is_anomaly'] == 1]
    normais = df[df['pred_is_anomaly'] == 0]

    # Reduz os normais para no máximo 5000 para não estourar a memória do JS
    if len(normais) > 5000:
        normais_amostra = normais.sample(n=5000, random_state=42)
    else:
        normais_amostra = normais
        
    df_plot = pd.concat([anomalias, normais_amostra])

    # 2. PLOTAGEM COM CORES PADRONIZADAS
    fig = px.histogram(
        df_plot, 
        x='anomaly_score', 
        color='pred_is_anomaly', 
        barmode='overlay',
        #color_discrete_map={False: '#1f77b4', True: '#ff4b4b'}, # Mesmas cores da timeline
        color_discrete_map={0: '#1f77b4', 1: '#ff4b4b',False: '#1f77b4', True: '#ff4b4b'},
        title="Distribuição dos Scores de Anomalia (Amostra Otimizada)",
        labels={'anomaly_score': 'Decision Score', 'count': 'Quantidade'}
    )
    
    # 3. FUNDO TRANSPARENTE (Deixa com visual de painel integrado)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )
    
    return fig

#@st.fragment
def generate_interactive_network(df, output_path="temp_graph.html"):
    """
    Grafo de Similaridade de Logs (Anomalias vs Normais):
    Nós vermelhos = Anômalos | Nós azuis = Normais
    """
    if df.empty:
        return None

    coluna_texto = 'Template' if 'Template' in df.columns else 'Event'

    # 1. Mais bolinhas: Aumentamos de 20 para 45 de cada tipo (Total de até 90 bolinhas na tela)
    top_anomalias = df[df['pred_is_anomaly'] == True][coluna_texto].value_counts().head(100)
    top_normais = df[df['pred_is_anomaly'] == False][coluna_texto].value_counts().head(100)

    nodes_info = []
    
    for texto, freq in top_anomalias.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'pred_is_anomaly': True})
        
    for texto, freq in top_normais.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'pred_is_anomaly': False})

    if not nodes_info:
        return None

    linhas_unicas = [n['texto'] for n in nodes_info]
    G = nx.Graph()

    # 2. Criar os Nós com Tamanho Controlado
    for i, info in enumerate(nodes_info):
        label_curto = textwrap.shorten(info['texto'], width=40, placeholder="...")
        
        cor = '#FF6B6B' if info['pred_is_anomaly'] else "#1BBB06" 
        status_txt = "🔴 ANOMALIA" if info['pred_is_anomaly'] else "🟢 NORMAL"
        
        # Crescimento suavizado (usando raiz quadrada **) para não ficar gigante
        # Tamanho base é 10.
        tamanho_calculado = 10 + (info['freq'] ** 0.5) * 1.5 
        
        # Limitamos o tamanho máximo da bolinha em 35 pixels (antes estava 300)
        tamanho = min(tamanho_calculado, 25)

        G.add_node(
            i, 
            label=label_curto, 
            title=f"{status_txt}\n\nLog Completo:\n{info['texto']}\n\nOcorrências: {info['freq']}", 
            size=tamanho, 
            color=cor 
        )

    # 3. Criar as conexões matemáticas (Mais visíveis)
    if len(linhas_unicas) >= 5:
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(linhas_unicas)
            matriz_similaridade = cosine_similarity(tfidf_matrix)

            for i in range(len(linhas_unicas)):
                for j in range(i + 1, len(linhas_unicas)):
                    sim = matriz_similaridade[i, j]
                    
                    # Reduzimos para 5% de similaridade para criar MAIS conexões
                    #if sim > 0.05:
                    #Aumentando para 20% de similaridade para reduzir a quantidade de linhas e deixar o grafo mais limpo
                    if sim > 0.20:  
                        # Multiplicador aumentado (de 5 para 8) para deixar as linhas mais gordinhas e visíveis
                        G.add_edge(i, j, weight=sim * 8, title=f"Similaridade: {sim:.0%}")
        except ValueError:
            pass

    # 4. Domando a Física do PyVis para o novo formato
    net = Network(height='450px', width='100%', bgcolor='#0E1117', font_color='white')
    net.from_nx(G)
    
    # Física aproximada: node_distance menor agrupa melhor as famílias de logs
    # damping mais alto faz elas pararem de "dançar" na tela mais rápido
    net.repulsion(node_distance=150, central_gravity=0.08, spring_length=150, damping=0.09)
    net.save_graph(output_path)
    
    return output_path


def plot_confusion_matrix_plotly(cm):
    """Gera uma Matriz de Confusão interativa e elegante."""
    # Inverte a matriz apenas para o visual ficar no padrão acadêmico
    z = cm[::-1] 
    x = ['Predito: Normal', 'Predito: Anomalia']
    y = ['Real: Anomalia', 'Real: Normal']
    
    # Criar o Heatmap (Mapa de calor)
    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, 
        colorscale='Blues', 
        showscale=True
    )
    
    fig.update_layout(
        title_text='Matriz de Confusão', 
        title_x=0.5,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    return fig

@st.cache_data(show_spinner="Calculando posições do grafo...")
def graph_spring_layout(df, output_path="temp_graph_spring.html"):
    if df.empty:
        return None

    coluna_texto = 'Template' if 'Template' in df.columns else 'Event'

    # 1. Preparação dos nós (mantida a sua lógica original)
    top_anomalias = df[df['pred_is_anomaly'] == True][coluna_texto].value_counts().head(100)
    top_normais = df[df['pred_is_anomaly'] == False][coluna_texto].value_counts().head(100)

    nodes_info = []
    
    for texto, freq in top_anomalias.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'pred_is_anomaly': 1})
        
    for texto, freq in top_normais.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'pred_is_anomaly': 0})

    if not nodes_info:
        return None

    linhas_unicas = [n['texto'] for n in nodes_info]
    G = nx.Graph()

    # 2. Criar os Nós
    for i, info in enumerate(nodes_info):
        label_curto = textwrap.shorten(info['texto'], width=40, placeholder="...")
        cor = '#FF6B6B' if info['pred_is_anomaly'] else "#1BBB06" 
        status_txt = "🔴 ANOMALIA" if info['pred_is_anomaly'] else "🟢 NORMAL"
        
        tamanho_calculado = 10 + (info['freq'] ** 0.5) * 1.5 
        tamanho = min(tamanho_calculado, 25)

        G.add_node(
            i, 
            label=label_curto, 
            title=f"{status_txt}\n\nLog Completo:\n{info['texto']}\n\nOcorrências: {info['freq']}", 
            size=tamanho, 
            color=cor 
        )

    # 3. Criar as conexões matemáticas
    if len(linhas_unicas) >= 5:
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(linhas_unicas)
            matriz_similaridade = cosine_similarity(tfidf_matrix)

            for i in range(len(linhas_unicas)):
                for j in range(i + 1, len(linhas_unicas)):
                    sim = matriz_similaridade[i, j]
                    #Realizado aumento do threshold de similaridade para 50% para reduzir a quantidade de linhas e deixar o grafo mais limpo
                    if sim > 0.50:
                        G.add_edge(i, j, weight=sim * 8, title=f"Similaridade: {sim:.0%}")
        except ValueError:
            pass

    # ==========================================
    # 4. INICIALIZAÇÃO DO PYVIS COM FÍSICA AVANÇADA
    # (O nx.spring_layout e posições manuais foram removidos)
    # ==========================================
    net = Network(height='450px', width='100%', bgcolor='#0E1117', font_color='white')
    net.from_nx(G)
    
    # 5. Configuração em JavaScript da física de partículas
    # O "avoidOverlap: 1" é o que impede que os rótulos se amassem
    physics_options = """
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -200, 
          "centralGravity": 0.01,       
          "springLength": 200,           
          "springConstant": 0.08,        
          "avoidOverlap": 1              
        },
        "minVelocity": 0.3,
        "solver": "forceAtlas2Based"
      }
    }
    """
    net.set_options(physics_options)

    # 6. Salva e gera o HTML
    net.save_graph(output_path)

    return output_path

def plot_mttd_mtti_historico(historico_path="resultados/historico_metricas.json"):
    """Gera um gráfico de linha comparando a evolução do MTTD e MTTI ao longo do tempo."""
    if not os.path.exists(historico_path):
        return None
        
    try:
        with open(historico_path, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            
        df_hist = pd.DataFrame(dados)
        
        if df_hist.empty or 'MTTD_Segundos' not in df_hist.columns:
            return None

        fig = px.line(
            df_hist, 
            x='Timestamp_Lote', 
            y=['MTTD_Segundos', 'MTTI_Segundos'],
            markers=True,
            title="Evolução do Tempo de Resposta (MTTD vs MTTI)",
            labels={
                'Timestamp_Lote': 'Horário do Lote', 
                'value': 'Tempo (Segundos)',
                'variable': 'Métrica'
            }
        )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            legend_title_text='Métrica RCA',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )
        return fig
        
    except Exception as e:
        print(f"Erro ao plotar histórico MTTD/MTTI: {e}")
        return None


import plotly.graph_objects as go

def plot_comparativo_antes_depois(df_resultado):
    """
    Gera um gráfico comparando a quantidade real de anomalias (Ground Truth)
    com a quantidade detectada pelo algoritmo (Predição).
    """
    # Restringe às linhas com rótulo real: fontes sem Ground Truth (ex.:
    # logs_appficticio) entram como NaN após o concat dos lotes e não devem
    # contar em nenhuma das duas barras, senão "Detectado" soma fontes que
    # "Real" não conta e a comparação fica desbalanceada.
    df_resultado = df_resultado.dropna(subset=['y_true_label'])

    # Contagem Real (Ground Truth)
    reais_normais = (df_resultado['y_true_label'] == 0).sum()
    reais_anomalias = (df_resultado['y_true_label'] == 1).sum()
    
    # Contagem Predita (Depois da execução do modelo)
    pred_normais = (df_resultado['pred_is_anomaly'] == 0).sum()
    pred_anomalias = (df_resultado['pred_is_anomaly'] == 1).sum()
    
    fig = go.Figure(data=[
        go.Bar(name='Real (Ground Truth)', x=['Normais', 'Anomalias'], y=[reais_normais, reais_anomalias], marker_color='#1f77b4'),
        go.Bar(name='Detectado pelo Modelo', x=['Normais', 'Anomalias'], y=[pred_normais, pred_anomalias], marker_color='#ff4b4b')
    ])
    
    fig.update_layout(
        title="Impacto do Algoritmo: Volume Real vs. Detectado",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        yaxis=dict(title='Quantidade de Logs', showgrid=True, gridcolor='#30363d')
    )
    return fig

def plot_metricas_destaque(precision, recall, f1):
    """
    Gera um gráfico de barras horizontais bem visível para as métricas matemáticas.
    """
    fig = go.Figure(go.Bar(
        x=[precision, recall, f1],
        y=['Precision', 'Recall', 'F1-Score'],
        orientation='h',
        text=[f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"],
        textposition='auto',
        textfont=dict(size=18, family="JetBrains Mono"),
        marker_color=['#58a6ff', '#d29922', '#238636']
    ))
    
    fig.update_layout(
        title="Métricas de Desempenho (RCA Pipeline)",
        xaxis=dict(range=[0, 1.1], title="Pontuação", showgrid=True, gridcolor='#30363d'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=350
    )
    return fig

def plot_walkforward_metricas_por_split(df_resultados):
    """
    Gera um gráfico de linha com F1/Precision/Recall/PR_AUC por split da
    avaliação walk-forward, com uma linha pontilhada marcando a média de
    cada métrica entre os splits.

    Espera um DataFrame com uma linha por split e (pelo menos algumas das)
    colunas 'split', 'F1_Score', 'Precision', 'Recall', 'PR_AUC' — o mesmo
    formato que avaliacao_walkforward.py já monta para o JSON de saída.
    """
    colunas_metricas = [c for c in ['F1_Score', 'Precision', 'Recall', 'PR_AUC'] if c in df_resultados.columns]
    if df_resultados.empty or not colunas_metricas or 'split' not in df_resultados.columns:
        return None

    cores = {'F1_Score': '#238636', 'Precision': '#58a6ff', 'Recall': '#d29922', 'PR_AUC': '#f778ba'}

    fig = go.Figure()
    for metrica in colunas_metricas:
        if df_resultados[metrica].notna().sum() == 0:
            continue

        fig.add_trace(go.Scatter(
            x=df_resultados['split'], y=df_resultados[metrica],
            mode='lines+markers', name=metrica,
            line=dict(color=cores.get(metrica, '#8b949e')),
            marker=dict(size=9)
        ))

        media = df_resultados[metrica].mean()
        fig.add_hline(
            y=media, line_dash="dot", line_color=cores.get(metrica, '#8b949e'), opacity=0.5,
            annotation_text=f"{metrica} médio: {media:.3f}", annotation_position="top left"
        )

    fig.update_layout(
        title="Avaliação Walk-Forward: Métricas por Split",
        xaxis=dict(title="Split (janela cronológica)", dtick=1, showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(title="Pontuação", range=[0, 1.05], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend_title_text='Métrica'
    )
    return fig

def plot_pr_curve_threshold(y_true, scores, threshold_usado=None,
                             titulo="Curva Precision-Recall e Sensibilidade ao Threshold",
                             label_threshold_usado="Threshold usado",
                             caminho_saida=None):
    """
    Evidência visual de duas partes, calculada SOMENTE a partir de arrays que
    o script chamador já produziu (y_true/scores/threshold reais da própria
    execução) — esta função não recalcula nada, só desenha:

      (a) Curva Precision-Recall (Recall x Precision), com a área sob a
          curva (PR-AUC) no título do painel.
      (b) Precision/Recall/F1 em função do threshold de decisão, com uma
          linha pontilhada cinza no melhor F1 teoricamente possível (maior
          F1 percorrendo TODOS os cortes da curva) e, se informado, uma
          linha vermelha tracejada no threshold efetivamente usado na
          classificação reportada — para mostrar visualmente se o corte
          usado está perto ou longe do ponto ótimo.

    Parâmetros
    ----------
    y_true : array-like binário (1 = anomalia real, 0 = normal).
    scores : array-like contínuo, mesma convenção usada no resto do
        pipeline (quanto MAIOR, mais anômalo — ou seja, -anomaly_score /
        -decision_function, nunca a coluna anomaly_score crua).
    threshold_usado : float opcional, no mesmo eixo de `scores`, marcando o
        corte realmente aplicado na classificação (df_resultado['pred_is_anomaly']).
    caminho_saida : caminho .png opcional; se informado, salva a figura ali
        (cria a pasta se preciso) além de retorná-la.

    Retorna a figura matplotlib (fig) para quem quiser plt.show()/ajustar.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)  # execução headless (scripts de linha de comando, sem display)
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recalls, precisions)

    # precision_recall_curve devolve um ponto a mais que thresholds (o ponto
    # final, sem corte associado) — descartamos esse último ponto de
    # precisions/recalls para casar o comprimento com thresholds.
    f1_scores = np.divide(
        2 * precisions[:-1] * recalls[:-1],
        precisions[:-1] + recalls[:-1],
        out=np.zeros_like(precisions[:-1]),
        where=(precisions[:-1] + recalls[:-1]) > 0,
    )
    idx_melhor_f1 = int(np.argmax(f1_scores)) if len(f1_scores) else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Curva Precision-Recall
    ax1.plot(recalls, precisions, color="#238636", linewidth=2)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title(f"Curva Precision-Recall (PR-AUC = {pr_auc:.4f})")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.3)

    # (b) Sensibilidade ao threshold
    ax2.plot(thresholds, precisions[:-1], label="Precision", color="#58a6ff")
    ax2.plot(thresholds, recalls[:-1], label="Recall", color="#d29922")
    ax2.plot(thresholds, f1_scores, label="F1-Score", color="#238636")
    if idx_melhor_f1 is not None:
        ax2.axvline(
            thresholds[idx_melhor_f1], color="gray", linestyle=":",
            label=f"Melhor F1 possível ({f1_scores[idx_melhor_f1]:.3f})"
        )
    if threshold_usado is not None:
        ax2.axvline(
            threshold_usado, color="#d62728", linestyle="--",
            label=f"{label_threshold_usado} ({threshold_usado:.4f})"
        )
    ax2.set_xlabel("Threshold de decisão (quanto maior, mais anômalo)")
    ax2.set_ylabel("Pontuação")
    ax2.set_title("Sensibilidade ao Threshold")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(alpha=0.3)

    fig.suptitle(titulo)
    fig.tight_layout()

    if caminho_saida:
        pasta = os.path.dirname(caminho_saida)
        if pasta:
            os.makedirs(pasta, exist_ok=True)
        fig.savefig(caminho_saida, dpi=150)
        print(f"   -> {caminho_saida} (PR-AUC={pr_auc:.4f})")

    plt.close(fig)
    return fig
