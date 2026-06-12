import os
import time
from datetime import date
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Importando seus módulos
import modules.parse_system as parse_system
import modules.preprocessor as preprocessor
import modules.anomaly_detector as anomaly_detector
import modules.visualizer as visualizer

st.set_page_config(layout="wide", page_title="Detecção de Anomalias - TCC")

# ==========================================
# MAPEAMENTO DE PASTAS DE LOGS
# ==========================================
PASTAS_DISPONIVEIS = [
    "docker/meus_logs",
    "logpai/Apache",  
    "logpai/Linux",
    "logpai/HDFS",
    "logpai/OpenSSH",   
    "logpai/Zookeeper",
    "minikube/k8s-chaos/logs"   
]
# ==========================================

def main():
    st.title("Anomaly Detection Dashboard for Logs 📊")
    
    # --- MENU LATERAL ---
    st.sidebar.header("1. Configurações de Monitoramento")
    
    # INTERRUPTOR DE TEMPO REAL
    auto_refresh = st.sidebar.toggle("⏱️ Atualização Automática (60s)", value=True, help="Desligue para poder marcar as caixinhas de validação sem a tela recarregar.")
    
    pastas_selecionadas = st.sidebar.multiselect(
        "Selecione as origens dos logs:",
        options=PASTAS_DISPONIVEIS,
        default=PASTAS_DISPONIVEIS 
    )

    st.sidebar.header("2. Inteligência Artificial")
    contamination = st.sidebar.slider(
        "Taxa de Contaminação (Anomalias)", 
        min_value=0.01, max_value=0.10, value=0.03, step=0.01
    )
    
    # --- PROCESSAMENTO AUTOMÁTICO ---
    # Agora o código roda direto, sem depender de botão!
    if len(pastas_selecionadas) == 0:
        st.warning("⚠️ Selecione pelo menos uma pasta de origem na barra lateral para iniciar o monitoramento.")
        return

    # Usamos o placeholder apenas para dar um feedback visual se os dados forem muito grandes
    status_text = st.empty()
    status_text.info("🔄 Lendo e processando logs atuais...")

    # PASSO 1: Leitura e Parse
    df_list = []
    for pasta in pastas_selecionadas:
        if os.path.exists(pasta):
            read_generic = parse_system.read_dir_to_temps(pasta)
            for path in read_generic:
                df_p = parse_system.automatic_drain_parse(path)
                if not df_p.empty:
                    df_p['Source_Folder'] = pasta
                    df_list.append(df_p)

    if not df_list:
        status_text.error("Nenhum dado válido extraído. Verifique as pastas.")
        return

    df_logs = pd.concat(df_list, ignore_index=True)

    # PASSO 2: Vetorização
    if 'Template' in df_logs.columns and 'Event' not in df_logs.columns:
        df_logs['Event'] = df_logs['Template']
        df_logs['Source'] = df_logs['Source_Folder'] 
        df_logs['Level'] = "INFO" 
        
    X_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_logs) 

    # PASSO 3: Detecção com Isolation Forest
    df_final, model = anomaly_detector.process_log_anomalies(df_logs, X_tfidf, contamination=contamination)

    today = date.today().strftime("%Y-%m-%d")
    output_name = f"resultado_tcc_{today}.csv"
    df_final.to_csv(output_name, index=False)

    status_text.empty() # Limpa a mensagem de "processando" quando acaba

    # --- EXIBIÇÃO DE MÉTRICAS E GRÁFICOS ---
    anomalias = df_final[df_final['is_anomaly'] == True]
    normais = df_final[df_final['is_anomaly'] == False]

    st.markdown("---")
    st.subheader("Resumo da Análise (Live)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Logs Processados", len(df_final))
    col2.metric("Anomalias Detectadas", len(anomalias), f"{contamination*100}%")
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

    st.markdown("### Grafo de Semelhança (Anomalias vs Normais)")
    html_graph = visualizer.generate_interactive_network(df_final)
    if html_graph and os.path.exists(html_graph):
        with open(html_graph, 'r', encoding='utf-8') as f:
            source_code = f.read()
        components.html(source_code, height=450)
    else:
        st.info("Não há anomalias suficientes para gerar o grafo de conexões.")

    # ==========================================
    # VALIDAÇÃO ESPECIALISTA (Precision @ 25%)
    # ==========================================
    st.markdown("---")
    st.subheader("Validação Especialista (Precision @ Top 85%)")
    
    if auto_refresh:
        st.warning("⚠️ **Atenção:** O painel está em modo 'Tempo Real'. Para auditar os falsos positivos na tabela abaixo sem a tela recarregar, desligue o interruptor no Menu Lateral.")

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
        
        st.info(f"**Resultado Interativo:** De {k_valor} logs críticos avaliados, o especialista validou {acertos} como falhas reais.")
        st.metric(f"Métrica Precision (Amostra de 85%)", f"{precisao_k:.1%}")
    else:
        st.success("Nenhuma anomalia para validar.")

    # ==========================================
    # O MOTOR DE TEMPO REAL (LOOP 120s)
    # ==========================================
    if auto_refresh:
        time.sleep(120)
        st.rerun()

if __name__ == "__main__":
    main()