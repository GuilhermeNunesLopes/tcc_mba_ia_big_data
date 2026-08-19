import os
import time
from datetime import date, datetime, time as dtime, timedelta
import pandas as pd
import streamlit as st
import json
import streamlit.components.v1 as components
import numpy as np

# Importando o módulo de visualização
import visualizer as visualizer

# Configuração de página com layout fluido real
st.set_page_config(
    layout="wide", 
    page_title="AIOPS platform | Anomaly Detection tool",
    initial_sidebar_state="collapsed"
)

PASTAS_DISPONIVEIS = ["logs_filtrados", "docker/meus_logs", "minikube/k8s-chaos/logs","experimento/test1","experimento/test2"]

# MUDANÇA: Exibe spinner ao carregar e indexar os Parquets
@st.cache_data(show_spinner="📥 Indexando dados de telemetria em memória...")
def carregar_dados_otimizado(caminho_arquivo, timestamp_modificacao):
    df = pd.read_parquet(caminho_arquivo)
    for col in ['Source_Folder', 'Level']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

@st.cache_data(show_spinner="🧩 Computando matriz de similaridade e topologia...")
def gerar_grafo_otimizado(df_selecionado, tipo="normal"):
    if tipo == "normal":
        return visualizer.generate_interactive_network(df_selecionado)
    return visualizer.graph_spring_layout(df_selecionado)

def format_duration(seconds):
    """
    Converte segundos brutos em escala humana (ms, s, m s, h m).
    """
    if seconds is None or np.isnan(seconds) or seconds < 0:
        return "N/A"
    
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    elif seconds < 3600.0:
        minutos = int(seconds // 60)
        segs = int(seconds % 60)
        return f"{minutos}m {segs}s"
    else:
        horas = int(seconds // 3600)
        minutos = int((seconds % 3600) // 60)
        return f"{horas}h {minutos}m"

def aplicar_tema_profissional(fig, tipo_grafico="scatter"):
    """
    Injeta o Design System de monitoramento nos gráficos do Plotly com alto contraste.
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#8b949e', family='Inter, sans-serif', size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=True, gridcolor='#30363d', gridwidth=1, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#30363d', gridwidth=1, zeroline=False),
        hoverlabel=dict(bgcolor="#161b22", font_size=13, font_family="Inter", bordercolor="#30363d")
    )
    
    if tipo_grafico == "scatter":
        for trace in fig.data:
            trace_name = str(trace.name).lower()
            if any(k in trace_name for k in ['1', 'true', 'anomalia']):
                trace.update(marker=dict(color='#FF0000', size=10, opacity=1.0))
            else:
                trace.update(marker=dict(color='#00FF00', size=5, opacity=0.6))

    elif tipo_grafico == "histogram":
        for trace in fig.data:
            trace_name = str(trace.name).lower()
            if any(k in trace_name for k in ['1', 'true', 'anomalia']):
                trace.update(marker_color='#FF4B4B', opacity=0.85)
            else:
                trace.update(marker_color='#1F77B4', opacity=0.5)

    elif tipo_grafico == "line_mttd":
        cores = ['#d29922', '#58a6ff']
        for i, trace in enumerate(fig.data):
            trace.update(line=dict(color=cores[i % len(cores)], width=3), marker=dict(size=6, color=cores[i % len(cores)]))

    elif tipo_grafico == "line_score":
        fig.for_each_trace(lambda t: t.update(line=dict(color='#238636', width=3), marker=dict(size=6, color='#238636')))
        
    return fig

def main():
    # ==========================================
    # CSS GRID SYSTEM E DESIGN
    # ==========================================
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-app: #0d1117;
            --bg-panel: #161b22;
            --border-color: #30363d;
            --text-muted: #8b949e;
            --text-main: #c9d1d9;
            --color-critical: #f85149;
            --color-warning: #d29922;
            --color-success: #238636;
            --color-info: #58a6ff;
        }

        .stApp { background-color: var(--bg-app); font-family: 'Inter', sans-serif !important; }
        
        /* Limpeza da UI nativa do Streamlit */
        header, footer, .stDeployButton { display: none !important; }
        
        /* Ajuste do container principal */
        .block-container { 
            padding: 70px 1.5rem 2rem 1.5rem !important; 
            max-width: 100% !important; 
        }

        .fixed-top-bar, .kpi-wrapper, .kpi-card, [data-testid="stForm"] {
            box-sizing: border-box !important;
        }

        /* TOP BAR Fixa */
        .fixed-top-bar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 55px;
            background-color: var(--bg-app);
            border-bottom: 1px solid var(--border-color);
            z-index: 9999 !important; /* Ajustado para não cobrir toasts */
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 24px;
            color: var(--text-main);
        }
        .brand-zone { display: flex; align-items: center; gap: 12px; font-weight: 600; font-size: 15px; letter-spacing: 0.5px; text-transform: uppercase; }
        .live-badge { background: var(--color-success); color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
        .status-zone { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-muted); font-weight: 500; }
        .status-dot { width: 8px; height: 8px; background: var(--color-success); border-radius: 50%; box-shadow: 0 0 8px var(--color-success); }

        /* COMMAND BAR (Filtros Fixos) */
        [data-testid="stForm"] {
            position: sticky;
            top: 55px;
            z-index: 9998 !important; /* Ajustado para ficar abaixo dos avisos */
            background-color: rgba(22, 27, 34, 0.95);
            backdrop-filter: blur(10px);
            border: none !important;
            border-bottom: 1px solid var(--border-color) !important;
            padding: 12px 24px 0 24px !important;
            margin-bottom: 16px !important;
            border-radius: 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        [data-testid="stToastContainer"], div[data-testid="stToast"] {
            z-index: 1000000 !important; /* Fica acima da Top Bar e do Form */
            top: 65px !important;        /* Empurra o aviso para baixo do menu fixo */
            right: 20px !important;
        }
        /* ESTILIZAÇÃO DO SPINNER E ALERTAS STREAMLIT */
        .stSpinner > div {
            border-top-color: var(--color-info) !important;
        }
        div[data-testid="stToast"] {
            background-color: var(--bg-panel) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-main) !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4) !important;
        }

        /* GRIDS E KPIS */
        .kpi-wrapper {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            width: 100%;
            margin-bottom: 16px;
        }
        
        .kpi-card {
            background-color: var(--bg-panel);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .kpi-title { color: var(--text-muted); font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
        .kpi-value { font-size: 1.8rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; line-height: 1.2; }

        .val-critical { color: var(--color-critical); }
        .val-success { color: var(--color-success); }
        .val-warning { color: var(--color-warning); } 
        .val-info { color: var(--color-info); }       
        .val-neutral { color: var(--text-main); }

        [data-testid="column"] {
            background-color: var(--bg-panel);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 16px;
        }
        
        h4, h5 {
            color: var(--text-main) !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
            margin: 0 0 16px 0 !important;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }

        div[data-testid="stDataFrameResizable"] {
            border: 1px solid var(--border-color) !important;
            border-radius: 6px !important;
        }
        div[data-testid="stDataFrameResizable"] table { 
            background-color: var(--bg-app) !important; 
        }
        div[data-testid="stDataFrameResizable"] th {
            background-color: var(--bg-panel) !important;
            color: var(--text-muted) !important;
            border-bottom: 1px solid var(--border-color) !important;
            font-size: 0.8rem !important;
            text-transform: uppercase;
        }
        div[data-testid="stDataFrameResizable"] td {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.8rem !important;
            color: var(--text-muted) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # GLOBAL NAVIGATION BAR FIXA
    # ==========================================
    st.markdown("""
        <div class="fixed-top-bar">
            <div class="brand-zone">
                <span class="live-badge">LIVE</span>
                <span>AIOPS platform | Anomaly Detection tool</span>
            </div>
            <div class="status-zone">
                <span class="status-dot"></span> Pipeline: Unsupervised ML
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # DATA INGESTION
    # ==========================================
    today = date.today().strftime("%Y-%m-%d")
    arquivo_dados = f"resultados/resultado_tcc_{today}.parquet"
    arquivo_metricas = "resultados/metricas_rca.json"

    metricas = {"Total_Incidentes": 0, "MTTD_Segundos": 0.0, "MTTI_Segundos": 0.0, "PR_AUC": 0.0}
    if os.path.exists(arquivo_metricas):
        try:
            with open(arquivo_metricas, "r", encoding="utf-8") as f:
                metricas.update(json.load(f))
        except: pass

    if not os.path.exists(arquivo_dados):
        st.markdown("<div style='padding: 60px; text-align: center; color: #8b949e; font-family: Inter;'>Engine de ML aguardando dados de telemetria...</div>", unsafe_allow_html=True)
        time.sleep(5)
        st.rerun()
        return

    tempo_atual_csv = os.path.getmtime(arquivo_dados)
    df_final = carregar_dados_otimizado(arquivo_dados, tempo_atual_csv)
    
    data_min = data_max = date.today()
    if 'Timestamp' in df_final.columns:
        df_final['Timestamp'] = pd.to_datetime(df_final['Timestamp'], errors='coerce').dt.tz_localize(None)
        df_final = df_final.dropna(subset=['Timestamp'])
        if not df_final.empty:
            data_min = df_final['Timestamp'].min().date()
            data_max = df_final['Timestamp'].max().date()

    # ==========================================
    # COMMAND BAR STICKY (Filtros Fixos)
    # ==========================================
    with st.form(key='global_filters'):
        # Expandimos para 7 colunas proporcionais para acomodar a Sensibilidade
        col_f1, col_f2, col_f3, col_f4, col_f5, col_f6, col_f7 = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        
        with col_f1:
            pastas_selecionadas = st.multiselect("Namespaces", options=PASTAS_DISPONIVEIS, default=PASTAS_DISPONIVEIS)
        with col_f2:
            algoritmo_selecionado = st.selectbox("Modelo", options=["iforest", "ocsvm"], format_func=lambda x: "iForest" if x == "iforest" else "OCSVM")
        with col_f3:
            reducao_selecionada = st.selectbox("Redução", options=["pca", "svd"], format_func=lambda x: "PCA" if x == "pca" else "SVD")
        with col_f4:
            datas_selecionadas = st.date_input("Data", value=(max(data_min, data_max - timedelta(days=30)), data_max))
        with col_f5:
            horas_selecionadas = st.slider("Timeframe", value=(dtime(0, 0), dtime(23, 59)), format="HH:mm")
        with col_f6:
            # NOVO: Slider de sensibilidade na 6ª coluna
            sensibilidade = st.slider("Alerta (%)", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
        with col_f7:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label='Aplicar Filtros', use_container_width=True)

    # LÓGICA AO CLICAR NO BOTÃO:
    if submit_button:
        tempo_inicial_parquet = os.path.getmtime(arquivo_dados) if os.path.exists(arquivo_dados) else 0

        # Salva o config.json enviando o valor da sensibilidade convertido para decimal (ex: 3.0% vira 0.03)
        with open("config.json", "w", encoding='utf-8') as f:
            json.dump({
                "pastas": pastas_selecionadas, 
                "taxa_contaminacao": sensibilidade / 100.0,  # <--- Injeta a Sensibilidade aqui
                "algoritmo": algoritmo_selecionado,
                "reducao": reducao_selecionada
            }, f)

        with st.spinner(f"⚙️ Re-treinando modelo ({algoritmo_selecionado.upper()} + {reducao_selecionada.upper()}) e processando logs..."):
            st.toast("⚡ Solicitando reprocessamento ao motor de ML...", icon="🚀")
            
            processado = False
            for _ in range(60): # Aguarda até 30 segundos
                time.sleep(0.5)
                if os.path.exists(arquivo_dados):
                    mod_atual = os.path.getmtime(arquivo_dados)
                    if mod_atual > tempo_inicial_parquet:
                        processado = True
                        break
            
            st.cache_data.clear()
            
            if processado:
                st.toast("✅ Dashboard atualizado com os novos dados!", icon="🎉")
            else:
                st.toast("⚠️ O motor demorou para responder. Verifique o terminal do main.py.", icon="⏳")
                
            time.sleep(0.5)
            st.rerun()

    if isinstance(datas_selecionadas, tuple) and len(datas_selecionadas) == 2:
        df_final = df_final[(df_final['Timestamp'] >= datetime.combine(datas_selecionadas[0], horas_selecionadas[0])) & 
                            (df_final['Timestamp'] <= datetime.combine(datas_selecionadas[1], horas_selecionadas[1]))]

    anomalias = df_final[df_final['pred_is_anomaly'] == 1]
    normais = df_final[df_final['pred_is_anomaly'] == 0]

    # ==========================================
    # RESPONSIVE KPI GRID (COM FORMATADOR DE TEMPO)
    # ==========================================
    pr_auc_str = f"{metricas.get('PR_AUC', 0):.1%}" if metricas.get('PR_AUC') else "N/A"
    
    mttd_humano = format_duration(metricas.get('MTTD_Segundos', 0))
    mtti_humano = format_duration(metricas.get('MTTI_Segundos', 0))

    html_kpis = f"""
    <div class="kpi-wrapper">
        <div class="kpi-card">
            <div class="kpi-title">Eventos Críticos (Alerts)</div>
            <div class="kpi-value val-critical">{len(anomalias):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">MTTD (Mean Time to Detect)</div>
            <div class="kpi-value val-warning">{mttd_humano}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">MTTI (Mean Time to Investigate)</div>
            <div class="kpi-value val-info">{mtti_humano}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Qualidade PR-AUC</div>
            <div class="kpi-value val-success">{pr_auc_str}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Logs Ingeridos (Throughput)</div>
            <div class="kpi-value val-neutral">{len(df_final):,}</div>
        </div>
    </div>
    """
    st.markdown(html_kpis, unsafe_allow_html=True)

    # ==========================================
    # ANALYTICS DASHBOARDS
    # ==========================================
    plotly_config = {'displayModeBar': False}

    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.markdown("<h5>Distribuição Temporal de Logs (Scoring)</h5>", unsafe_allow_html=True)
        fig_timeline = visualizer.plot_anomaly_timeline_plotly(df_final)
        fig_timeline = aplicar_tema_profissional(fig_timeline, "scatter")
        st.plotly_chart(fig_timeline, use_container_width=True, config=plotly_config)

    with col_graf2:
        st.markdown("<h5>Concentração de Decision Scores</h5>", unsafe_allow_html=True)
        fig_dist = visualizer.plot_anomaly_distribution_plotly(df_final)
        fig_dist = aplicar_tema_profissional(fig_dist, "histogram")
        st.plotly_chart(fig_dist, use_container_width=True, config=plotly_config)
    
    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        st.markdown("<h5>Coesão de Agrupamento (Silhouette Trend)</h5>", unsafe_allow_html=True)
        fig_hist = visualizer.plot_metricas_historico()
        if fig_hist:
            fig_hist = aplicar_tema_profissional(fig_hist, "line_score")
            st.plotly_chart(fig_hist, use_container_width=True, config=plotly_config)
        else:
            st.markdown("<div style='padding:40px; color:#8b949e; text-align:center;'>Acumulando baselines...</div>", unsafe_allow_html=True)
            
    with col_hist2:
        st.markdown("<h5>Degradação de SLA (MTTD vs MTTI)</h5>", unsafe_allow_html=True)
        fig_mttd = visualizer.plot_mttd_mtti_historico()
        if fig_mttd:
            fig_mttd = aplicar_tema_profissional(fig_mttd, "line_mttd")
            st.plotly_chart(fig_mttd, use_container_width=True, config=plotly_config)
        else:
            st.markdown("<div style='padding:40px; color:#8b949e; text-align:center;'>Aguardando encerramento de incidentes...</div>", unsafe_allow_html=True)

    # ==========================================
    # ROOT CAUSE TOPOGRAPHY
    # ==========================================
    with st.container():
        st.markdown("""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin-bottom: 16px;">
            <h5 style="margin:0; border:none; padding:0;">Mapeamento de Causa Raiz (Topologia de Grafos)</h5>
        </div>
        """, unsafe_allow_html=True)
        
        aba_normal, aba_spring = st.tabs(["Layout Hierárquico", "Física Dinâmica (ForceAtlas2)"])
        with aba_normal:
            if st.button("Executar Traçado de Correlação", key="btn_n"):
                with st.spinner("Mapeando vizinhança vetorial..."):
                    html_graph = gerar_grafo_otimizado(df_final, tipo="normal")
                    if html_graph and os.path.exists(html_graph):
                        with open(html_graph, 'r', encoding='utf-8') as f:
                            components.html(f.read(), height=500, scrolling=False)
        with aba_spring:
            if st.button("Executar Traçado Vetorial", key="btn_s"):
                with st.spinner("Calculando atração topológica..."):
                    html_graph = gerar_grafo_otimizado(df_final, tipo="spring")
                    if html_graph and os.path.exists(html_graph):
                        with open(html_graph, 'r', encoding='utf-8') as f:
                            components.html(f.read(), height=500, scrolling=False)

    # ==========================================
    # LOG EXPLORER & TRIAGEM DE ALERTAS
    # ==========================================
    st.markdown("""
    <div style="margin-top: 24px;">
        <h5 style="border-bottom: 1px solid #30363d; padding-bottom: 8px;">Log Explorer (Alert Triage - P@K)</h5>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. VISÃO NUMÉRICA COMPARATIVA: Normais vs Anômalos
    total_logs_count = len(df_final)
    total_anomalias_count = len(anomalias)
    total_normais_count = len(normais)
    pct_anomalias = (total_anomalias_count / total_logs_count * 100) if total_logs_count > 0 else 0

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; text-align: center;">
            <span style="color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight: 600;">Logs Normais</span>
            <h3 style="color: #238636; margin: 4px 0 0 0; font-family: 'JetBrains Mono', monospace; border: none; padding: 0;">{total_normais_count:,}</h3>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; text-align: center;">
            <span style="color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight: 600;">Logs Anômalos</span>
            <h3 style="color: #f85149; margin: 4px 0 0 0; font-family: 'JetBrains Mono', monospace; border: none; padding: 0;">{total_anomalias_count:,}</h3>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; text-align: center;">
            <span style="color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight: 600;">Proporção de Anomalia</span>
            <h3 style="color: #d29922; margin: 4px 0 0 0; font-family: 'JetBrains Mono', monospace; border: none; padding: 0;">{pct_anomalias:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # 2. TABELA DE TRIAGEM COM BUSCA E LOGS REAIS
    if not anomalias.empty:
        # ---- NOVO: LAYOUT DE BUSCA E SLIDER ----
        col_search, col_slider = st.columns([3, 1])
        with col_search:
            busca_texto = st.text_input("🔍 Buscar termo nos logs anômalos (ex: Timeout, Failed, Error):", "")
        with col_slider:
            k_selecionado = st.slider("Amostragem Crítica (Top N):", min_value=5, max_value=100, value=20, step=5)
            
        k_valor = min(k_selecionado, len(anomalias))
        
        # Filtra os top K logs com menores decision scores (mais anômalos)
        top_k_logs = anomalias.nsmallest(k_valor, 'anomaly_score').copy()
        
        # ---- NOVO: APLICAÇÃO DO FILTRO DE BUSCA ----
        if busca_texto:
            top_k_logs = top_k_logs[top_k_logs['Raw_Log'].str.contains(busca_texto, case=False, na=False)]
            
        # Verifica se sobrou algo após a busca
        if top_k_logs.empty:
            st.warning(f"Nenhuma anomalia encontrada contendo o termo '{busca_texto}'.")
        else:
            top_k_logs['Triage (True Positive)'] = False 
            
            # ---- NOVO: GARANTE A COLUNA CLUSTER_ID (DBSCAN) ----
            if 'cluster_id' not in top_k_logs.columns:
                top_k_logs['cluster_id'] = "Isolado"
            else:
                top_k_logs['cluster_id'] = top_k_logs['cluster_id'].fillna("Isolado").astype(str)

            # ---- NOVO: GARANTE A COLUNA DE EXPLICABILIDADE (fallback para parquets antigos) ----
            if 'Termos_Explicativos' not in top_k_logs.columns:
                top_k_logs['Termos_Explicativos'] = "N/D (rode com a versão atual do pipeline)"
            else:
                top_k_logs['Termos_Explicativos'] = top_k_logs['Termos_Explicativos'].fillna("sem termos distintivos")

            # Exibe o data editor
            df_editado = st.data_editor(
                top_k_logs[['Triage (True Positive)', 'Source_Folder', 'cluster_id', 'Raw_Log', 'anomaly_score', 'Termos_Explicativos']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Triage (True Positive)": st.column_config.CheckboxColumn(
                        "Confirmar Incidente?",
                        help="Marque para validar se este log é uma anomalia real",
                        default=False
                    ),
                    "Source_Folder": st.column_config.TextColumn("Origem", width="small"),
                    "cluster_id": st.column_config.TextColumn("RCA Cluster", width="small"),
                    "Raw_Log": st.column_config.TextColumn("Log Real (Texto Completo)", width="large"),
                    "anomaly_score": st.column_config.NumberColumn("Decision Score", format="%.4f", width="small"),
                    "Termos_Explicativos": st.column_config.TextColumn(
                        "Por que foi sinalizado?",
                        help="Termos TF-IDF de maior peso presentes neste log específico",
                        width="large"
                    )
                },
                disabled=['Source_Folder', 'cluster_id', 'Raw_Log', 'anomaly_score', 'Termos_Explicativos']
            )
            
            acertos = df_editado['Triage (True Positive)'].sum()
            # Ajusta o denominador para o tamanho real da tabela (caso a busca tenha filtrado resultados)
            tamanho_tabela_exibida = len(top_k_logs)
            precisao_k = acertos / tamanho_tabela_exibida if tamanho_tabela_exibida > 0 else 0
            
            st.markdown(f"<div style='margin-top:10px; color:#8b949e; font-size:13px; font-family: Inter;'><strong>SLA Report:</strong> Operador marcou {acertos} anomalias como incidentes reais dentre {tamanho_tabela_exibida} exibidas. Precisão de Alerta da query atual: <span style='color:#58a6ff; font-weight: 600;'>{precisao_k:.1%}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding: 20px 0; color: #238636; font-size:14px; font-family: Inter;'>✓ Estado Nominal do Sistema. A query não retornou logs na zona de alerta.</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()