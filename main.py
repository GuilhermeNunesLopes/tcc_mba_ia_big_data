import os
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
# Coloque aqui as duas pastas do loghub e a sua do docker
PASTAS_DISPONIVEIS = [
    "docker/meus_logs",
    "logpai/Apache",  
    "logpai/Linux",
    "logpai/OpenSSH",
    "logpai/Spark"   
]
# ==========================================

def main():
    st.title("Monitoramento de Logs e Detecção de Anomalias 🚀")
    
    # --- MENU LATERAL ---
    st.sidebar.header("1. Fonte dos Dados")
    
    # O usuário pode marcar ou desmarcar quais pastas quer analisar
    pastas_selecionadas = st.sidebar.multiselect(
        "Selecione as origens dos logs:",
        options=PASTAS_DISPONIVEIS,
        default=PASTAS_DISPONIVEIS # Por padrão, já vem com todas marcadas
    )

    st.sidebar.header("2. Configurações do Modelo")
    contamination = st.sidebar.slider(
        "Taxa de Contaminação (Anomalias)", 
        min_value=0.01, max_value=0.10, value=0.02, step=0.01
    )
    
    # --- BOTÃO DE PROCESSAMENTO ---
    # Só habilita o botão se tiver pelo menos uma pasta selecionada
    if st.button("Processar Logs Agora", type="primary", disabled=len(pastas_selecionadas)==0):
        
        # PASSO 1: Leitura e Parse de MÚLTIPLAS pastas
        with st.spinner("Lendo e parseando logs com Drain3..."):
            df_list = []
            
            # Faz um loop por todas as pastas que você selecionou no menu
            for pasta in pastas_selecionadas:
                if not os.path.exists(pasta):
                    st.warning(f"⚠️ Pasta não encontrada, ignorando: {pasta}")
                    continue
                    
                read_generic = parse_system.read_dir_to_temps(pasta)
                
                for path in read_generic:
                    df_p = parse_system.automatic_drain_parse(path)
                    if not df_p.empty:
                        # Adiciona uma coluna para sabermos de onde esse log veio!
                        df_p['Source_Folder'] = pasta
                        df_list.append(df_p)

            if not df_list:
                st.error("Nenhum dado válido extraído. Verifique as pastas.")
                return

            df_logs = pd.concat(df_list, ignore_index=True)
            st.success(f"Logs parseados com sucesso! ({len(df_logs)} linhas totais)")

        # PASSO 2: Vetorização
        with st.spinner("Vetorizando e detectando anomalias..."):
            if 'Template' in df_logs.columns and 'Event' not in df_logs.columns:
                df_logs['Event'] = df_logs['Template']
                # Agora usamos o nome da pasta como a "Fonte" do log em vez de um nome fixo
                df_logs['Source'] = df_logs['Source_Folder'] 
                df_logs['Level'] = "INFO" 
                
            X_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_logs) 

            # PASSO 3: Detecção com Isolation Forest
            df_final, model = anomaly_detector.process_log_anomalies(df_logs, X_tfidf, contamination=contamination)

            today = date.today().strftime("%Y-%m-%d")
            output_name = f"resultado_tcc_{today}.csv"
            df_final.to_csv(output_name, index=False)

        # --- EXIBIÇÃO DE MÉTRICAS E GRÁFICOS ---
        anomalias = df_final[df_final['is_anomaly'] == True]
        normais = df_final[df_final['is_anomaly'] == False]

        st.markdown("---")
        st.subheader("Resumo da Análise")
        
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

        st.markdown("### Grafo de Semelhança (Anomalias)")
        with st.spinner("Calculando similaridade e gerando grafo interativo..."):
            html_graph = visualizer.generate_interactive_network(df_final)
            if html_graph and os.path.exists(html_graph):
                with open(html_graph, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                components.html(source_code, height=450)
            else:
                st.info("Não há anomalias suficientes para gerar o grafo de conexões.")

        st.markdown("---")
        st.subheader("Amostra dos Logs Anômalos")
        if not anomalias.empty:
            # Exibe a origem da pasta na tabela para você saber de onde veio o erro
            st.dataframe(anomalias[['Source_Folder', 'Cluster_ID', 'Template', 'anomaly_score']].head(50), use_container_width=True)
            
            with open(output_name, "rb") as file:
                st.download_button(
                    label="📥 Baixar CSV Completo para o TCC",
                    data=file,
                    file_name=output_name,
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()