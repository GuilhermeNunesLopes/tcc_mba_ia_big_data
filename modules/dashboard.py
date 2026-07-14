import os
import time
from datetime import date, datetime, time as dtime, timedelta
import pandas as pd
import streamlit as st
import json
import streamlit.components.v1 as components

# Importando APENAS o módulo de visualização
import visualizer as visualizer

# A configuração da página deve ser SEMPRE o primeiro comando Streamlit (fora do main)
st.set_page_config(layout="wide", page_title="Detecção de Anomalias - TCC")

PASTAS_DISPONIVEIS = [
    # "docker/meus_logs",
    # "logpai/Apache",  
    # "logpai/Linux",
    # "logpai/HDFS",
    # "logpai/OpenSSH",   
    # "logpai/Zookeeper",
    "minikube/k8s-chaos/logs"   
]

@st.cache_data(show_spinner=False)
def carregar_dados_otimizado(caminho_arquivo, timestamp_modificacao):
    df = pd.read_parquet(caminho_arquivo)
    if 'Source_Folder' in df.columns:
        df['Source_Folder'] = df['Source_Folder'].astype('category')
    if 'Level' in df.columns:
        df['Level'] = df['Level'].astype('category')
    return df

@st.cache_data(show_spinner="Calculando posições do grafo...")
def gerar_grafo_otimizado(df_selecionado, tipo="normal"):
    if tipo == "normal":
        return visualizer.generate_interactive_network(df_selecionado)
    else:
        return visualizer.graph_spring_layout(df_selecionado)

def main():
    # ==========================================
    # 1. CSS E TÍTULO PRINCIPAL (Devem ser os primeiros a renderizar)
    # ==========================================
    st.markdown("""
        <style>
        /* 1. Fundo do Painel com Gradiente Radial (Foco no centro) */
        .stApp {
            background-color: #0A0E17;
            background-image: radial-gradient(circle at 50% 0%, #151A28 0%, #0A0E17 70%);
        }
        
        /* 2. Cartões de Métricas (Efeito Neon e Vidro) */
        div[data-testid="metric-container"] {
            background-color: rgba(15, 20, 30, 0.6);
            border: 1px solid #1F2937;
            border-left: 4px solid #00F2FE;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.05);
            padding: 15px 20px;
            border-radius: 6px;
            backdrop-filter: blur(4px);
        }
        
        /* Título das métricas com fonte de terminal */
        div[data-testid="metric-container"] label {
            color: #8B949E !important;
            font-family: 'Courier New', Courier, monospace !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        
        /* Valores das métricas brilhantes */
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #FFFFFF;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
            font-family: 'Courier New', Courier, monospace !important;
        }

        /* 3. Estilização da Tabela de Validação (Estilo Terminal Hacker) */
        div[data-testid="stDataFrameResizable"] {
            border: 1px solid #30363D;
            border-radius: 8px;
            background-color: #0D1117;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
        }

        /* 4. Esconder a "Sujeira" do Streamlit para parecer um App nativo */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container { padding-top: 2rem !important; }
        </style>
    """, unsafe_allow_html=True)

    st.title("Anomaly Detection Dashboard for Logs 📊")

    # COMPONENTE JS: STATUS AO VIVO E RELÓGIO (Mantive o seu código intacto)
    components.html(
        """
        <div style="display: flex; align-items: center; background-color: rgba(15, 20, 30, 0.8); border: 1px solid #1F2937; padding: 10px 20px; border-radius: 5px; color: #00F2FE; font-family: 'Courier New', monospace; font-size: 14px; font-weight: bold; text-transform: uppercase; letter-spacing: 2px;">
            <div style="width: 12px; height: 12px; background-color: #FF4B4B; border-radius: 50%; margin-right: 15px; box-shadow: 0 0 10px #FF4B4B; animation: blink 1.5s infinite;"></div>
            <span>SISTEMA DE DETECÇÃO ATIVO</span>
            <span id="digital-clock" style="margin-left: auto; color: #FFFFFF; text-shadow: 0 0 5px rgba(255,255,255,0.5);"></span>
        </div>
        <script>
            setInterval(() => {
                const now = new Date();
                document.getElementById('digital-clock').innerText = now.toLocaleTimeString('pt-BR');
            }, 1000);
        </script>
        <style>
            @keyframes blink { 
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.3; transform: scale(0.9); }
                100% { opacity: 1; transform: scale(1); }
            }
        </style>
        """, 
        height=60 
    )

    # ==========================================
    # 2. MÉTRICAS DE RCA (Lidas logo abaixo do título e do relógio)
    # ==========================================
    st.subheader("⏱️ Desempenho do RCA (Tempo Real)")
    arquivo_metricas = "resultados/metricas_rca.json"

    if os.path.exists(arquivo_metricas):
        try:
            with open(arquivo_metricas, "r", encoding="utf-8") as f:
                metricas = json.load(f)
                
            col1, col2, col3 = st.columns(3)
            col1.metric("Incidentes Processados", metricas.get("Total_Incidentes", 0))
            col2.metric("MTTD (Detecção)", f"{metricas.get('MTTD_Segundos', 0)}s")
            col3.metric("MTTI (Investigação)", f"{metricas.get('MTTI_Segundos', 0)}s")
            
        except json.JSONDecodeError:
            pass
    else:
        st.info("🔄 Aguardando o processamento do primeiro lote de logs para calcular as métricas de MTTD e MTTI...")

    # ==========================================
    # 3. LEITURA DO PARQUET E CONTINUAÇÃO DO FLUXO
    # ==========================================
    today = date.today().strftime("%Y-%m-%d")
    arquivo_dados = f"resultados/resultado_tcc_{today}.parquet"
    
    if not os.path.exists(arquivo_dados):
        st.warning("⚠️ Aguardando dados do Motor de Processamento...")
        st.info("Certifique-se de que o arquivo `main.py` está rodando no terminal para gerar o arquivo.")
        time.sleep(5)
        st.rerun()
        return

    # 3. LÊ O CSV IMEDIATAMENTE (Precisamos dele para saber as datas)
    tempo_atual_csv = os.path.getmtime(arquivo_dados)
    df_final = carregar_dados_otimizado(arquivo_dados, tempo_atual_csv)
    
    # ... AQUI CONTINUA O SEU CÓDIGO ORIGINAL (Prepara as variáveis de data, Sidebar, Filtros, etc.) ...
    # Prepara as variáveis de data com valores padrão de segurança
    data_min = date.today()
    data_max = date.today()
    coluna_tempo = 'Timestamp' 
    
    if coluna_tempo in df_final.columns:
        df_final[coluna_tempo] = pd.to_datetime(df_final[coluna_tempo], errors='coerce')
        df_final = df_final.dropna(subset=[coluna_tempo])
        if not df_final.empty:
            data_min = df_final[coluna_tempo].min().date()
            data_max = df_final[coluna_tempo].max().date()
    
    
    
    # ==========================================
    # CÁLCULO DOS ÚLTIMOS 30 DIAS
    # ==========================================
    # Calcula 30 dias atrás a partir da data mais recente do dataset.
    # O uso do 'max()' garante que não vamos tentar selecionar uma data mais antiga que o próprio 'data_min'.
    data_inicio_padrao = max(data_min, data_max - timedelta(days=30))
    
    # Define a tupla padrão para o calendário
    if data_min == data_max:
        valor_calendario = data_max
    else:
        valor_calendario = (data_inicio_padrao, data_max)
    # ==========================================


    with st.sidebar.form(key='filtro_form'):
        # --- 4. MENU LATERAL E FORMULÁRIO ---
        st.sidebar.header("Configurações Gerais")
        auto_refresh = st.sidebar.toggle("⏱️ Atualizar Tela (60s)", value=False)
        
        pastas_selecionadas = st.sidebar.multiselect(
            "Origens dos logs:",
            options=PASTAS_DISPONIVEIS,
            default=PASTAS_DISPONIVEIS 
        )

        st.header("Filtros Temporais & Contaminação")
        
        contamination = st.slider(
            "Taxa de Contaminação (Anomalias)", 
            min_value=0.01, max_value=0.10, value=0.03, step=0.01
        )
        
        # O calendário agora sempre abrirá focando na janela de 30 dias (ou menos, se não houver dados suficientes)
        datas_selecionadas = st.date_input(
            "Período (Dias):",
            value=valor_calendario,
            min_value=data_min,
            max_value=data_max
        )
        
        horas_selecionadas = st.slider(
            "Intervalo de Horário:",
            value=(dtime(0, 0), dtime(23, 59)), 
            format="HH:mm" 
        )
        
        # O botão que trava o recálculo automático!
        submit_button = st.form_submit_button(label='Aplicar Filtros 🚀')

    # ==========================================
    # 5. SALVA AS CONFIGURAÇÕES E SINCRONIZA COM O MOTOR
    # ==========================================
    
    # 1. SÓ SALVA O JSON SE O BOTÃO FOR CLICADO
    if submit_button:
        config_data = {
            "pastas": pastas_selecionadas,
            "taxa_contaminacao": contamination
        }
        with open("config.json", "w", encoding='utf-8') as f:
            json.dump(config_data, f)
            
        # Zera a contagem de espera toda vez que aplicamos um filtro novo
        st.session_state.espera_motor = 0

    # 2. VERIFICAÇÃO DE SINCRONIA (A tela espera o Motor terminar)
    if os.path.exists("config.json"):
        tempo_json = os.path.getmtime("config.json")
        tempo_dados = os.path.getmtime(arquivo_dados) # Usa a data do Parquet atual
        
        # Se o JSON foi modificado DEPOIS do arquivo de dados, o Motor ainda está trabalhando
        if tempo_json > tempo_dados:
            if 'espera_motor' not in st.session_state:
                st.session_state.espera_motor = 0
                
            if st.session_state.espera_motor < 15: # Aumentei as tentativas para 15
                st.session_state.espera_motor += 1
                st.warning(f"⏳ **IA analisando novos dados...** (Tentativa {st.session_state.espera_motor}/15)")
                with st.spinner("O Motor está gerando o modelo. Aguarde..."):
                    # OTIMIZAÇÃO: Checa a cada 3 segundos em vez de 30!
                    time.sleep(3) 
                    st.rerun()
            else:
                st.error("⚠️ O Motor demorou muito para atualizar o arquivo. Verifique o terminal do backend.")
                st.session_state.espera_motor = 0 
        else:
            # Sincronizado!
            st.session_state.espera_motor = 0

    # ==========================================
    # 6. FILTRA OS DADOS (Com base no que o usuário escolheu no formulário)
    # ==========================================
    if isinstance(datas_selecionadas, tuple) and len(datas_selecionadas) == 2:
        data_inicio, data_fim = datas_selecionadas
        hora_inicio, hora_fim = horas_selecionadas
        
        filtro_inicio = datetime.combine(data_inicio, hora_inicio)
        filtro_fim = datetime.combine(data_fim, hora_fim)
        
        mask = (df_final[coluna_tempo] >= filtro_inicio) & (df_final[coluna_tempo] <= filtro_fim)
        df_final = df_final.loc[mask]

        
    anomalias = df_final[df_final['pred_is_anomaly'] == True]
    normais = df_final[df_final['pred_is_anomaly'] == False]

    # ==========================================
    # 7. RENDERIZAÇÃO PROGRESSIVA
    # ==========================================
    st.markdown("---")
    st.subheader("Resumo da Análise (Live)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Logs Filtrados", len(df_final))
    col2.metric("Anomalias Detectadas", len(anomalias))
    col3.metric("Logs Normais", len(normais))

    st.markdown("---")
    st.subheader("Visualizações")

    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        fig_timeline = visualizer.plot_anomaly_timeline_plotly(df_final)
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col_graf2:
        fig_dist = visualizer.plot_anomaly_distribution_plotly(df_final)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown("### Grafo de Semelhança de Logs")
    
    aba_normal, aba_spring = st.tabs(["Layout Normal", "Layout Spring (Física)"])
    
    with aba_normal:
        if st.button("Gerar Grafo Normal 🕸️", key="btn_normal"):
            with st.spinner("Calculando física do grafo..."):
                html_graph = gerar_grafo_otimizado(df_final, tipo="normal")
                if html_graph and os.path.exists(html_graph):
                    with open(html_graph, 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=470, scrolling=False)
                else:
                    st.info("Não há anomalias suficientes para este grafo.")

    with aba_spring:
        if st.button("Gerar Grafo Spring ⚛️", key="btn_spring"):
            with st.spinner("Calculando física vetorial..."):
                html_graph_2 = gerar_grafo_otimizado(df_final, tipo="spring")
                if html_graph_2 and os.path.exists(html_graph_2):
                    with open(html_graph_2, 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=470, scrolling=False)
                else:
                    st.info("Não há anomalias suficientes para o grafo Spring.")

    st.markdown("---")
    st.subheader("Validação Especialista (Precision @ Top 85%)")
    
    if auto_refresh:
        st.warning("⚠️ O painel está em modo 'Tempo Real'. Para auditar os falsos positivos, desligue o interruptor no Menu.")

    if not anomalias.empty:
        k_valor = max(1, int(len(anomalias) * 0.85))
        top_k_logs = anomalias.nsmallest(k_valor, 'anomaly_score').copy()
        top_k_logs['É Falha Real?'] = True
        
        colunas_mostrar = ['É Falha Real?', 'Source_Folder', 'Template', 'anomaly_score']
        df_editado = st.data_editor(
            top_k_logs[colunas_mostrar],
            hide_index=True,
            use_container_width=True,
            disabled=['Source_Folder', 'Template', 'anomaly_score'] 
        )
        
        acertos = df_editado['É Falha Real?'].sum()
        precisao_k = acertos / k_valor
        
        st.info(f"**Resultado:** De **{k_valor}** logs avaliados, o especialista validou **{acertos}** como falhas reais.")
        st.metric(f"Métrica Precision", f"{precisao_k:.1%}")
    else:
        st.success("Nenhuma anomalia para validar neste período.")

    if auto_refresh:
        time.sleep(60) 
        st.rerun()

if __name__ == "__main__":
    main()