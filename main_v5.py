import os
import time
from datetime import date
import pandas as pd
import subprocess
import json
import platform
import atexit
import numpy as np

# Importando apenas os módulos de processamento e IA (sem o dashboard)
import modules.parse_system as parse_system
import pipeline as pipeline
from modules.mttd_mtti import RCA_MetricsTracker
from modules.config_pastas import PASTAS_DISPONIVEIS
from sklearn.model_selection import train_test_split

# Iniciando o Tracking do MTTD e MTTI
tracker = RCA_MetricsTracker()

# ==========================================
# MAPEAMENTO DE PASTAS E CONFIGURAÇÕES
# ==========================================
def ler_configuracoes():
    """Lê o arquivo de configuração gerado pelo Streamlit."""
    try:
        with open("config.json", "r", encoding='utf-8') as f:
            config = json.load(f)
            # Retorna: pastas, contaminação, ALGORITMO e REDUÇÃO (padrão: pca)
            return (
                config.get("pastas", []), 
                config.get("taxa_contaminacao", "auto"), 
                config.get("algoritmo", "iforest"),
                config.get("reducao", "pca")  # <--- Nova chave para PCA vs SVD
            )
    except (FileNotFoundError, json.JSONDecodeError):
        pastas_padrao = PASTAS_DISPONIVEIS
        return pastas_padrao, "auto", "iforest", "pca"


def coletar_logs(pastas_ativas):
    """Varre as pastas configuradas e retorna o DataFrame concatenado de logs parseados pelo Drain3."""
    df_list = []
    for pasta in pastas_ativas:
        if os.path.exists(pasta):
            read_generic = parse_system.read_dir_to_temps(pasta)
            for path in read_generic:
                nome_da_fonte = os.path.basename(pasta)
                gerador_lotes = parse_system.automatic_drain_parse(path, nome_fonte=nome_da_fonte, tamanho_lote=100000)

                for df_lote in gerador_lotes:
                    if not df_lote.empty:
                        df_lote['Source_Folder'] = pasta
                        df_list.append(df_lote)

    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)


def processar_logs_em_lote():
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 Iniciando varredura de logs...")

    pastas_ativas, taxa_contaminacao_ativa, algoritmo_ativo, reducao_ativa = ler_configuracoes()

    if not pastas_ativas:
        print(f"[{time.strftime('%H:%M:%S')}] ⏸️ Nenhuma pasta selecionada no painel. Aguardando...")
        return

    lote_id = f"batch_{int(time.time())}"

    os.makedirs("resultados", exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")
    caminho_parquet = f"resultados/resultado_tcc_{today}.parquet"

    print(f"[{time.strftime('%H:%M:%S')}] 📥 Coletando e parseando logs (Drain3)...")
    df_logs = coletar_logs(pastas_ativas)

    if df_logs.empty:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Nenhum dado válido encontrado nas pastas.")
        if os.path.exists(caminho_parquet):
            os.utime(caminho_parquet, None)
        return

    print(f"[{time.strftime('%H:%M:%S')}] 🛡️ Aplicando Whitelist e Engenharia de Features Temporais...")
    df_logs = pipeline.preprocessar_logs_brutos(df_logs)

    if df_logs.empty:
        print(f"[{time.strftime('%H:%M:%S')}] ⏸️ Todos os logs deste lote eram conhecidos (Whitelist). Aguardando novos logs...")
        return

    # Divisão Cronológica dos dados (80% Treino / 20% Teste)
    print(f"[{time.strftime('%H:%M:%S')}] 🔀 Dividindo os dados CRONOLOGICAMENTE (80% Passado / 20% Futuro)...")
    df_train, df_test = train_test_split(df_logs, test_size=0.2, shuffle=False)

    print(f"[{time.strftime('%H:%M:%S')}] 🧠 Treinando modelo {algoritmo_ativo.upper()} com Redução {reducao_ativa.upper()}...")
    print(f"[{time.strftime('%H:%M:%S')}] 🎯 Aplicando inferência, extraindo métricas e explicações...")

    resultado = pipeline.treinar_e_avaliar(
        df_train, df_test,
        taxa_contaminacao_ativa=taxa_contaminacao_ativa,
        algoritmo_ativo=algoritmo_ativo,
        reducao_ativa=reducao_ativa,
        top_n_termos=5,
        calcular_rca=True,
    )

    df_resultado = resultado["df_resultado"]
    metricas_ml = resultado["metricas_ml"]
    y_verdadeiro = resultado["y_verdadeiro"]
    score_silhueta = resultado["score_silhueta"]

    if y_verdadeiro is None:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Modo Puramente Não Supervisionado (sem coluna de rótulo real).")
    elif metricas_ml:
        print(f"[{time.strftime('%H:%M:%S')}] 📊 Métricas de generalização (threshold congelado do treino): "
              f"F1={metricas_ml.get('F1_Score', 0):.4f} | PR_AUC={metricas_ml.get('PR_AUC', 0):.4f}")

    if score_silhueta is not None:
        print(f"[{time.strftime('%H:%M:%S')}] 📐 Silhouette Score do Incidente: {score_silhueta:.4f}")

    # ---- RASTREAMENTO DE INCIDENTES (MTTD/MTTI) POR CLUSTER REAL ----
    # Antes: um único tracker.start_injection(lote_id)/mark_detected/
    # mark_isolated por LOTE inteiro — Total_Incidentes ficava sempre 1 e
    # o t0 era o wall-clock do início do processamento, não o timestamp
    # real do incidente (causa raiz dos outliers de ~20 anos no MTTD
    # histórico). Agora: um incidente por cluster de anomalia que o DBSCAN
    # encontrou dentro de pipeline.treinar_e_avaliar (df_resultado já traz
    # 'cluster_id' quando há RCA), com t0 = timestamp real do log mais
    # antigo daquele cluster.
    incidentes_do_lote = []
    if 'cluster_id' in df_resultado.columns:
        clusters_reais = df_resultado.loc[
            df_resultado['pred_is_anomaly'] == 1, 'cluster_id'
        ].dropna()
        clusters_reais = clusters_reais[clusters_reais != -1].unique()

        for cid in clusters_reais:
            incident_id = f"{lote_id}_cluster{int(cid)}"
            linhas_cluster = df_resultado[df_resultado['cluster_id'] == cid]
            t0_real = linhas_cluster['Timestamp'].min().timestamp()

            tracker.start_injection(incident_id, t0=t0_real)
            tracker.mark_detected(incident_id)   # detecção já concluída (scoring)
            tracker.mark_isolated(incident_id)   # RCA (DBSCAN) já concluída junto
            incidentes_do_lote.append(incident_id)

    if not incidentes_do_lote:
        # Sem cluster real (RCA não rodou, ou só ruído/-1): mantém o
        # comportamento antigo, 1 "incidente" = o lote inteiro, para não
        # perder visibilidade de lotes sem RCA aplicável.
        tracker.start_injection(lote_id)
        tracker.mark_detected(lote_id)
        tracker.mark_isolated(lote_id)

    # Limpeza final de nulos
    df_resultado = df_resultado.dropna(subset=['Raw_Log'])
    df_resultado['Raw_Log'] = df_resultado['Raw_Log'].astype(str).str.strip()
    df_resultado = df_resultado[df_resultado['Raw_Log'] != ""]

    df_resultado.to_parquet(caminho_parquet, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Processamento concluído! Salvo em: {caminho_parquet}")

    print(f"[{time.strftime('%H:%M:%S')}] 🕸️ Correlação topológica em Grafos disponível "
          f"({len(incidentes_do_lote) or 1} incidente(s) rastreado(s) neste lote).")

    # Consolidação das Métricas
    resultados_metricas = tracker.calculate_results()
    tracker.clear_batch() # Zera o dicionário para o próximo lote de logs
    
    if score_silhueta is not None:
        resultados_metricas["Silhouette_Score"] = round(float(score_silhueta), 4)

    if metricas_ml:
        resultados_metricas.update(metricas_ml)

    resultados_metricas["Timestamp_Lote"] = time.strftime('%Y-%m-%d %H:%M:%S')
    resultados_metricas["Tecnica_Reducao"] = reducao_ativa.upper()

    print(f"[{time.strftime('%H:%M:%S')}] 📊 Métricas do Lote: {resultados_metricas}")

    os.makedirs("resultados", exist_ok=True)

    with open("resultados/metricas_rca.json", "w", encoding="utf-8") as f:
        json.dump(resultados_metricas, f)

    historico_path = "resultados/historico_metricas.json"
    historico_dados = []

    if os.path.exists(historico_path):
        try:
            with open(historico_path, "r", encoding="utf-8") as f:
                historico_dados = json.load(f)
        except json.JSONDecodeError:
            pass

    historico_dados.append(resultados_metricas)
    with open(historico_path, "w", encoding="utf-8") as f:
        json.dump(historico_dados, f)


# Variável global para guardar o processo do dashboard
processo_dashboard = None

def limpar_processos_antigos():
    """Mata qualquer processo fantasma do Streamlit antes de iniciar um novo."""
    sistema = platform.system()
    try:
        if sistema == "Windows":
            os.system("taskkill /F /IM streamlit.exe >nul 2>&1")
        else:
            os.system("pkill -f 'streamlit' >/dev/null 2>&1")
    except Exception:
        pass

def fechar_dashboard_atual():
    """Garante que o dashboard atual feche se o main.py fechar."""
    global processo_dashboard
    if processo_dashboard is not None:
        processo_dashboard.terminate()
        processo_dashboard.wait()

atexit.register(fechar_dashboard_atual)

if __name__ == "__main__":
    print("🧹 Limpando processos fantasmas antigos...")
    limpar_processos_antigos()
    time.sleep(1) 
    
    print("🚀 Iniciando o Sistema de Detecção de Anomalias (v4)...")
    print("🖥️ Abrindo o Dashboard no navegador (Sempre na porta 8501)...")
    
    comando = ["streamlit", "run", "modules/dashboard.py", "--server.port", "8501"]
    processo_dashboard = subprocess.Popen(comando)
    
    time.sleep(3) 
    
    print("\n⚙️ Iniciando Motor de Processamento de Logs (Background)...")
    print("⚠️ Pressione CTRL+C no terminal para encerrar o motor e o painel.")
    print("-" * 50)
    
    try:
        ultimo_json_modificado = 0
        
        while True:
            processar_logs_em_lote()
            print("⏳ Aguardando novos lotes ou alteração de filtro na tela...\n")
            
            if os.path.exists("config.json"):
                ultimo_json_modificado = os.path.getmtime("config.json")
            
            # Loop responsivo de 1 segundo para ler o config.json instantaneamente
            for _ in range(200):
                time.sleep(1)
                if os.path.exists("config.json"):
                    modificacao_atual = os.path.getmtime("config.json")
                    if modificacao_atual > ultimo_json_modificado:
                        print("\n🔔 Nova configuração detectada! Acordando o motor imediatamente...")
                        break 
            
    except KeyboardInterrupt:
        print("\n🛑 Encerrando o sistema a pedido do usuário...")
        print("✅ Motor e Dashboard encerrados com sucesso.")
        
    except Exception as e:
        print(f"\n⚠️ Erro inesperado no motor: {e}")
        time.sleep(60)
