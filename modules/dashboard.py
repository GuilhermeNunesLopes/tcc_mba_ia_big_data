import os
import time
from datetime import date, datetime, time as dtime
import pandas as pd
import streamlit as st
import json
import streamlit.components.v1 as components

# Importando APENAS o módulo de visualização
import visualizer as visualizer

st.set_page_config(layout="wide", page_title="Detecção de Anomalias - TCC")

PASTAS_DISPONIVEIS = [
    "docker/meus_logs",
    "logpai/Apache",  
    "logpai/Linux",
    "logpai/HDFS",
    "logpai/OpenSSH",   
    "logpai/Zookeeper",
    "minikube/k8s-chaos/logs"   
]

@st.cache_data(show_spinner=False)
def carregar_dados_otimizado(caminho_arquivo, timestamp_modificacao):
    df = pd.read_csv(caminho_arquivo)
    
    # Converte textos repetitivos para categorias (ocupa muito menos memória)
    if 'Source_Folder' in df.columns:
        df['Source_Folder'] = df['Source_Folder'].astype('category')
    if 'Level' in df.columns:
        df['Level'] = df['Level'].astype('category')
        
    return df

@st.cache_data(show_spinner="Calculando posições do grafo...")
def gerar_grafo_otimizado(df_selecionado, tipo="normal"):
    """Gera o grafo interativo e guarda na memória para não recalcular a física à toa"""
    if tipo == "normal":
        return visualizer.generate_interactive_network(df_selecionado)
    else:
        return visualizer.graph_spring_layout(df_selecionado)

def main():
    st.title("Anomaly Detection Dashboard for Logs 📊")
    
    # 1. IDENTIFICA O ARQUIVO DO DIA
    today = date.today().strftime("%Y-%m-%d")
    arquivo_dados = f"resultados/resultado_tcc_{today}.csv"
    
    # 2. VERIFICA SE O ARQUIVO EXISTE ANTES DE TENTAR LER
    if not os.path.exists(arquivo_dados):
        st.warning("⚠️ Aguardando dados do Motor de Processamento...")
        st.info(f"Certifique-se de que o arquivo `main.py` está rodando no terminal para gerar o arquivo.")
        time.sleep(5)
        st.rerun()
        return

    # 3. LÊ O CSV IMEDIATAMENTE (Precisamos dele para saber as datas)
    tempo_atual_csv = os.path.getmtime(arquivo_dados)
    df_final = carregar_dados_otimizado(arquivo_dados, tempo_atual_csv)
    
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
    
    # --- 4. MENU LATERAL E FORMULÁRIO ---
    st.sidebar.header("1. Configurações Gerais")
    auto_refresh = st.sidebar.toggle("⏱️ Atualizar Tela (60s)", value=True)
    
    pastas_selecionadas = st.sidebar.multiselect(
        "Origens dos logs:",
        options=PASTAS_DISPONIVEIS,
        default=PASTAS_DISPONIVEIS 
    )
    
    with st.sidebar.form(key='filtro_form'):
        st.header("2. IA e Filtros Temporais 🕰️")
        
        contamination = st.slider(
            "Taxa de Contaminação (Anomalias)", 
            min_value=0.01, max_value=0.10, value=0.03, step=0.01
        )
        
        valor_calendario = data_min if data_min == data_max else (data_min, data_max)
        
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
    config_data = {
        "pastas": pastas_selecionadas,
        "taxa_contaminacao": contamination
    }
    with open("config.json", "w", encoding='utf-8') as f:
        json.dump(config_data, f)
        
    if os.path.exists("config.json"):
        tempo_json = os.path.getmtime("config.json")
        if tempo_json > tempo_atual_csv:
            if 'espera_motor' not in st.session_state:
                st.session_state.espera_motor = 0
                
            if st.session_state.espera_motor < 10:
                st.session_state.espera_motor += 1
                st.warning(f"⏳ **Recalculando a IA...** (Tentativa {st.session_state.espera_motor}/10)")
                with st.spinner("O Motor está gerando os novos dados. Aguarde..."):
                    time.sleep(3)
                    st.rerun()
            else:
                st.error("⚠️ O Motor não atualizou o CSV a tempo.")
                st.session_state.espera_motor = 0 
        else:
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
        
    anomalias = df_final[df_final['is_anomaly'] == True]
    normais = df_final[df_final['is_anomaly'] == False]

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
        html_graph = gerar_grafo_otimizado(df_final, tipo="normal")
        if html_graph and os.path.exists(html_graph):
            with open(html_graph, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=470, scrolling=False)
        else:
            st.info("Não há anomalias suficientes para este grafo.")

    with aba_spring:
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