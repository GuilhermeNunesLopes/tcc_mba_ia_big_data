import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import plotly.figure_factory as ff
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
        color_discrete_map={False: '#4D96FF', True: '#FF6B6B'}, # Mantendo o padrão de cores
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
    if len(linhas_unicas) >= 2:
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

    # 4. Domando a Física do PyVis para o novo formato
    net = Network(height='450px', width='100%', bgcolor='#222222', font_color='white')
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