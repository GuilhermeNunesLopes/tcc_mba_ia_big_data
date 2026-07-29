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
import os
import json

def plot_anomaly_timeline_plotly(df):
    """Gera uma linha do tempo super otimizada com cores de alto contraste."""
    
    #anomalias = df[df['pred_is_anomaly'] == True].copy()
    #normais = df[df['pred_is_anomaly'] == False].copy()
    anomalias = df[df['pred_is_anomaly'] == 1].copy()
    normais = df[df['pred_is_anomaly'] == 0].copy()

    if len(normais) > 5000:
        normais = normais.sample(n=5000, random_state=42)
        
    df_plot = pd.concat([normais, anomalias])
    
    # Adiciona uma coluna para forçar as anomalias a serem bolinhas MAIORES no gráfico
    df_plot['tamanho_ponto'] = df_plot['pred_is_anomaly'].apply(lambda x: 12 if x else 5)
    
    tem_timestamp = 'Timestamp' in df_plot.columns
    x_col = 'Timestamp' if tem_timestamp else df_plot.index
    x_label = 'Tempo (Hora do Log)' if tem_timestamp else 'Sequência dos Logs'
    
    hover_cols = []
    if 'Template' in df_plot.columns: hover_cols.append('Template')
    if 'Source_Folder' in df_plot.columns: hover_cols.append('Source_Folder')

    fig = px.scatter(
        df_plot, 
        x=x_col, 
        y='anomaly_score', 
        color='pred_is_anomaly',
        # Cores Fortes: Verde brilhante para normal, Vermelho Alerta para anomalias
        color_discrete_map={False: '#00FF00', True: '#FF0000'}, 
        size='tamanho_ponto', # Aplica a diferença de tamanho
        title="Linha do Tempo de Detecção de Anomalias",
        labels={x_col: x_label, 'anomaly_score': 'Decision Score (Gravidade)'},
        hover_data=hover_cols,
        render_mode='webgl'
    )
    
    # Estilização Hacker/SRE (Fundo escuro nativo)
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
        color_discrete_map={False: '#1f77b4', True: '#ff4b4b'}, # Mesmas cores da timeline
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
        
        # Verifica se o arquivo tem os dados necessários
        if df_hist.empty or 'MTTD_Segundos' not in df_hist.columns:
            return None
            
        fig = px.line(
            df_hist, 
            x='Timestamp_Lote', 
            y=['MTTD_Segundos', 'MTTI_Segundos'], # Passando as duas métricas juntas
            markers=True,
            title="Evolução do Tempo de Resposta a Incidentes (RCA)",
            labels={
                'Timestamp_Lote': 'Horário do Lote', 
                'value': 'Tempo em Segundos',
                'variable': 'Métrica'
            }
        )
        
        # Estilizando as linhas (MTTD em Amarelo, MTTI em Laranja)
        fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))
        
        # Fundo transparente e adequação ao tema
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            legend_title_text='Fases do RCA',
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        )
        return fig
        
    except Exception as e:
        print(f"Erro ao plotar histórico MTTD/MTTI: {e}")
        return None