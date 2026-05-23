import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

def plot_anomaly_timeline_plotly(df):
    """Gera uma linha do tempo interativa."""
    # Cria a figura base com Plotly Express
    fig = px.scatter(
        df, 
        x=df.index, 
        y='anomaly_score', 
        color='is_anomaly',
        color_discrete_map={False: 'blue', True: 'red'},
        title="Linha do Tempo de Detecção de Anomalias",
        labels={'index': 'Sequência dos Logs', 'anomaly_score': 'Decision Score'}
    )
    
    # Adiciona uma linha conectando os pontos para mostrar a tendência
    fig.add_trace(go.Scatter(
        x=df.index, y=df['anomaly_score'], 
        mode='lines', 
        line=dict(color='blue', width=1, dash='dot'),
        showlegend=False,
        opacity=0.3
    ))
    
    return fig

def plot_anomaly_distribution_plotly(df):
    """Gera um histograma interativo."""
    fig = px.histogram(
        df, 
        x='anomaly_score', 
        color='is_anomaly', 
        barmode='overlay',
        color_discrete_map={False: 'blue', True: 'red'},
        title="Distribuição dos Scores de Anomalia"
    )
    return fig

def generate_interactive_network(df, output_path="temp_graph.html"):
    """
    Grafo de Similaridade de Logs (Anomalias vs Normais):
    Nós vermelhos = Anômalos | Nós azuis = Normais
    """
    if df.empty:
        return None

    coluna_texto = 'Template' if 'Template' in df.columns else 'Event'

    # 1. Separar Normais e Anômalos 
    # Pegamos os Top 20 de cada para não travar o navegador com milhares de bolinhas
    top_anomalias = df[df['is_anomaly'] == True][coluna_texto].value_counts().head(20)
    top_normais = df[df['is_anomaly'] == False][coluna_texto].value_counts().head(20)

    # Cria uma lista unificada com as informações de cada nó
    nodes_info = []
    
    for texto, freq in top_anomalias.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'is_anomaly': True})
        
    for texto, freq in top_normais.items():
        nodes_info.append({'texto': str(texto), 'freq': freq, 'is_anomaly': False})

    if not nodes_info:
        return None

    linhas_unicas = [n['texto'] for n in nodes_info]
    G = nx.Graph()

    # 2. Criar os Nós com Cores Diferentes
    for i, info in enumerate(nodes_info):
        label_curto = textwrap.shorten(info['texto'], width=40, placeholder="...")
        
        # Define a cor com base no status (Vermelho para anomalia, Azul para normal)
        cor = '#FF6B6B' if info['is_anomaly'] else '#4D96FF' 
        status_txt = "🔴 ANOMALIA" if info['is_anomaly'] else "🔵 NORMAL"
        
        # Atenuamos o crescimento do tamanho da bolinha porque logs normais podem ter milhares de ocorrências
        tamanho = 15 + (info['freq'] * 0.1) if not info['is_anomaly'] else 15 + (info['freq'] * 2)
        # Limita o tamanho máximo para a bolinha não engolir a tela
        tamanho = min(tamanho, 100)

        G.add_node(
            i, 
            label=label_curto, 
            title=f"{status_txt}\n\nLog Completo:\n{info['texto']}\n\nOcorrências: {info['freq']}", 
            size=tamanho, 
            color=cor 
        )

    # 3. Criar as conexões matemáticas (Se houver 2 ou mais nós)
    if len(linhas_unicas) >= 2:
        # try/except previne um erro do TF-IDF caso os logs tenham apenas palavras muito curtas/estranhas
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(linhas_unicas)
            matriz_similaridade = cosine_similarity(tfidf_matrix)

            for i in range(len(linhas_unicas)):
                for j in range(i + 1, len(linhas_unicas)):
                    sim = matriz_similaridade[i, j]
                    
                    # Conecta os logs que tiverem mais de 10% de semelhança matemática
                    if sim > 0.10:
                        G.add_edge(i, j, weight=sim * 5, title=f"Similaridade: {sim:.0%}")
        except ValueError:
            pass

    # 4. Domando a Física do PyVis
    net = Network(height='450px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(G)
    
    # Física mais afastada (node_distance) para acomodar os dois grupos
    net.repulsion(node_distance=300, central_gravity=0.05, spring_length=250)
    net.save_graph(output_path)
    
    return output_path