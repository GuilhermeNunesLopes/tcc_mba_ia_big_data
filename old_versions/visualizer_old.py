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

def plot_anomaly_timeline_plotly(df):
    """Gera uma linha do tempo super otimizada para o navegador."""
    
    # 1. DOWNSAMPLING INTELIGENTE (O Segredo da Leveza)
    anomalias = df[df['is_anomaly'] == True]
    normais = df[df['is_anomaly'] == False]
    
    # Se houver mais de 3000 logs normais, pega uma amostra aleatória para não travar a tela
    if len(normais) > 3000:
        normais_amostra = normais.sample(n=3000, random_state=42)
    else:
        normais_amostra = normais
        
    # Junta de novo para plotar (Anomalias completas + Amostra de normais)
    df_plot = pd.concat([anomalias, normais_amostra])
    
    # Prepara as colunas (mesma lógica que fizemos antes)
    tem_timestamp = 'Timestamp' in df_plot.columns
    x_col = 'Timestamp' if tem_timestamp else df_plot.index
    x_label = 'Tempo (Hora do Log)' if tem_timestamp else 'Sequência dos Logs'
    
    hover_cols = []
    if 'Template' in df_plot.columns: hover_cols.append('Template')
    if 'Source_Folder' in df_plot.columns: hover_cols.append('Source_Folder')

    # Cria a figura base usando o df_plot (que é muito menor e mais rápido)
    fig = px.scatter(
        df_plot, 
        x=x_col, 
        y='anomaly_score', 
        color='pred_is_anomaly',
        # Mapeamento duplo (int e bool) para garantir contraste total: Verde vs Vermelho Alerta
        color_discrete_map={
            0: '#00FF66', 1: '#FF2A2A',
            False: '#00FF66', True: '#FF2A2A'
        }, 
        size='tamanho_ponto',
        title="Linha do Tempo de Detecção de Anomalias",
        labels={x_col: x_label, 'anomaly_score': 'Decision Score (Gravidade)'},
        hover_data=hover_cols,
        render_mode='webgl'
    )
    
    if tem_timestamp:
        df_sorted = df_plot.sort_values(by='Timestamp')
        line_x = df_sorted['Timestamp']
        line_y = df_sorted['anomaly_score']
    else:
        line_x = df_plot.index
        line_y = df_plot['anomaly_score']
        
    fig.add_trace(go.Scatter(
        x=line_x, 
        y=line_y, 
        mode='lines', 
        line=dict(color='#1f77b4', width=1, dash='dot'),
        showlegend=False,
        opacity=0.2
    ))
    
    # Estilização do Fundo do Gráfico (Deixa ele transparente para casar com o Streamlit)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )
    
    if tem_timestamp:
        fig.update_xaxes(rangeslider_visible=True)
        
    return fig

def plot_anomaly_distribution_plotly(df):
    """Gera um histograma interativo e super otimizado para a Web."""
    # 1. DOWNSAMPLING PARA O HISTOGRAMA
    anomalias = df[df['is_anomaly'] == True]
    normais = df[df['is_anomaly'] == False]
    
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
        color_discrete_map={
            0: '#1f77b4', 1: '#ff4b4b',
            False: '#1f77b4', True: '#ff4b4b'
        }, 
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
    top_anomalias = df[df['is_anomaly'] == True][coluna_texto].value_counts().head(100)
    top_normais = df[df['is_anomaly'] == False][coluna_texto].value_counts().head(100)

    nodes_info = []
    
    for texto, freq in top_anomalias.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'is_anomaly': True})
        
    for texto, freq in top_normais.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'is_anomaly': False})

    if not nodes_info:
        return None

    linhas_unicas = [n['texto'] for n in nodes_info]
    G = nx.Graph()

    # 2. Criar os Nós com Tamanho Controlado
    for i, info in enumerate(nodes_info):
        label_curto = textwrap.shorten(info['texto'], width=40, placeholder="...")
        
        cor = '#FF6B6B' if info['is_anomaly'] else "#1BBB06" 
        status_txt = "🔴 ANOMALIA" if info['is_anomaly'] else "🟢 NORMAL"
        
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
                    #Aumentando para 10% de similaridade para reduzir a quantidade de linhas e deixar o grafo mais limpo
                    if sim > 0.10:  
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
def graph_spring_layout(df,output_path="temp_graph_spring.html"):
    if df.empty:
        return None

    coluna_texto = 'Template' if 'Template' in df.columns else 'Event'

    # 1. Mais bolinhas: Aumentamos de 20 para 45 de cada tipo (Total de até 90 bolinhas na tela)
    top_anomalias = df[df['is_anomaly'] == True][coluna_texto].value_counts().head(100)
    top_normais = df[df['is_anomaly'] == False][coluna_texto].value_counts().head(100)

    nodes_info = []
    
    for texto, freq in top_anomalias.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'is_anomaly': True})
        
    for texto, freq in top_normais.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'is_anomaly': False})

    if not nodes_info:
        return None

    linhas_unicas = [n['texto'] for n in nodes_info]
    G = nx.Graph()

    # 2. Criar os Nós com Tamanho Controlado
    for i, info in enumerate(nodes_info):
        label_curto = textwrap.shorten(info['texto'], width=40, placeholder="...")
        
        cor = '#FF6B6B' if info['is_anomaly'] else "#1BBB06" 
        status_txt = "🔴 ANOMALIA" if info['is_anomaly'] else "🟢 NORMAL"
        
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
                    if sim > 0.05:
                        # Multiplicador aumentado (de 5 para 8) para deixar as linhas mais gordinhas e visíveis
                        G.add_edge(i, j, weight=sim * 8, title=f"Similaridade: {sim:.0%}")
        except ValueError:
            pass

    """Gera um layout de grafo usando o algoritmo de força de mola (spring layout)."""
    pos = nx.spring_layout(G, k=0.5, seed=42)  # Seed para reprodutibilidade

    for node in G.nodes():
        G.nodes[node]['x'] = pos[node][0] * 350  # Escala para melhor visualização
        G.nodes[node]['y'] = pos[node][1] * 350

    # 5. Inicializa o Pyvis
    net = Network(height='450px', width='100%', bgcolor='#0E1117', font_color='white')

    net.from_nx(G)
    # 6. Turn off live physics simulation to prevent the graph from re-adjusting
    net.toggle_physics(False)
 
    # 8. Domando a Física do PyVis para o novo formato
    net.save_graph(output_path)

    return output_path