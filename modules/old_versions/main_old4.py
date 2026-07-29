import os
import time
from datetime import date
import pandas as pd
import subprocess
import json
import platform
import atexit

# Importando apenas os módulos de processamento e IA
import modules.parse_system as parse_system
import modules.preprocessor as preprocessor
import modules.anomaly_detector as anomaly_detector
from modules.mttd_mtti import RCA_MetricsTracker
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Iniciando o Tracking do MTTI E MTTR
tracker = RCA_MetricsTracker()

# ==========================================
# MAPEAMENTO DE PASTAS DE LOGS
# ==========================================
def ler_configuracoes():
    """Lê o arquivo de configuração gerado pelo Streamlit."""
    try:
        with open("config.json", "r", encoding='utf-8') as f:
            config = json.load(f)
            # Valor de contaminação ajustado para 0.05 / mudei para auto para ver se temos uma melhora
            return config.get("pastas", []), config.get("taxa_contaminacao", "auto") 
    except (FileNotFoundError, json.JSONDecodeError):
        pastas_padrao = [
            "minikube/k8s-chaos/logs",
            "docker/meus_logs"
        ] 
        return pastas_padrao, "auto"

def processar_logs_em_lote():
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 Iniciando varredura de logs...")
    
    pastas_ativas, taxa_contaminacao_ativa = ler_configuracoes()

    if not pastas_ativas:
        print(f"[{time.strftime('%H:%M:%S')}] ⏸️ Nenhuma pasta selecionada no painel. Aguardando...")
        return
    
    lote_id = f"batch_{int(time.time())}"
    tracker.start_injection(lote_id)

    df_list = []
    for pasta in pastas_ativas:
        if os.path.exists(pasta):
            read_generic = parse_system.read_dir_to_temps(pasta)
            for path in read_generic:
                df_p = parse_system.automatic_drain_parse(path)
                if not df_p.empty:
                    df_p['Source_Folder'] = pasta
                    df_list.append(df_p)

    os.makedirs("resultados", exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")
    caminho_parquet = f"resultados/resultado_tcc_{today}.parquet"

    if not df_list:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Nenhum dado válido encontrado nas pastas.")
        if os.path.exists(caminho_parquet):
            os.utime(caminho_parquet, None) 
        return

    df_logs = pd.concat(df_list, ignore_index=True)

    if 'Template' in df_logs.columns and 'Event' not in df_logs.columns:
        df_logs['Event'] = df_logs['Template']
        df_logs['Source'] = df_logs['Source_Folder'] 
        df_logs['Level'] = "INFO"
        
    # ==========================================
    # IMPLEMENTAÇÃO DA WHITELIST (Filtro de Ruído)
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 🛡️ Aplicando Whitelist para descartar falsos positivos conhecidos...")
    
    # Adicione aqui as palavras-chave, serviços ou fragmentos de texto 
    # que você SABE que não são anomalias no seu ambiente Minikube/K8s.
    termos_whitelist = [
        "kube-proxy", 
        "healthcheck", 
        "get /healthz", 
        "ping",
        "connection closed by peer",
        "liveness probe",
        "readiness probe"
    ]
    
    # Cria uma máscara booleana vazia
    mask_ignorar = pd.Series(False, index=df_logs.index)
    
    # Procura cada termo da whitelist no texto original do log (Raw_Log)
    # case=False garante que ele vai pegar ignorando maiúsculas/minúsculas
    for termo in termos_whitelist:
        if 'Raw_Log' in df_logs.columns:
            mask_ignorar = mask_ignorar | df_logs['Raw_Log'].str.contains(termo, case=False, na=False, regex=False)
        elif 'Event' in df_logs.columns: # Fallback caso Raw_Log não exista
            mask_ignorar = mask_ignorar | df_logs['Event'].str.contains(termo, case=False, na=False, regex=False)
            
    # Remove as linhas que caíram na whitelist mantendo apenas o restante
    qtd_antes = len(df_logs)
    df_logs = df_logs[~mask_ignorar].copy()
    qtd_depois = len(df_logs)
    
    print(f"[{time.strftime('%H:%M:%S')}] 📉 Whitelist descartou {qtd_antes - qtd_depois} logs rotineiros.")

    # Se a whitelist filtrou TUDO, o lote encerra aqui para não quebrar a IA
    if df_logs.empty:
        print(f"[{time.strftime('%H:%M:%S')}] ⏸️ Todos os logs deste lote eram conhecidos (Whitelist). Aguardando novos logs...")
        return
    # ==========================================

    # O código original segue daqui em diante com os logs "limpos"
    matriz_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_logs)
    
    # ==========================================
    # NOVA LÓGICA DE TREINO E TESTE
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 🔀 Dividindo os dados (80% Treino / 20% Teste)...")
    df_train, df_test, X_train_tfidf, X_test_tfidf = train_test_split(
        df_logs, matriz_tfidf, test_size=0.2, random_state=42
    )

    # Ajuste T0: Tenta capturar a hora real do erro mais antigo deste lote para métricas precisas
    #if not df_test.empty and 'Timestamp' in df_test.columns:
    #    ts_min = pd.to_datetime(df_test['Timestamp'], errors='coerce').min()
    #    if pd.notnull(ts_min):
    #        tracker.incidents[lote_id]['t0'] = ts_min.timestamp()

    # ==========================================
    # REDUÇÃO DE DIMENSIONALIDADE (SVD)
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 📉 Reduzindo dimensionalidade (TruncatedSVD)...")
    
    X_train, modelo_svd = preprocessor.apply_truncated_svd(X_train_tfidf, svd_model=None, n_components=30)
    X_test, _ = preprocessor.apply_truncated_svd(X_test_tfidf, svd_model=modelo_svd)

    print(f"[{time.strftime('%H:%M:%S')}] 🧠 Treinando modelo Isolation Forest (Fase 1)...")
    _, modelo_treinado = anomaly_detector.process_log_anomalies(
        df_original=df_train, 
        X_tfidf=X_train, 
        contamination=taxa_contaminacao_ativa,
        model=None 
    )
    
    print(f"[{time.strftime('%H:%M:%S')}] 🎯 Aplicando inferência e extraindo métricas (Fase 2)...")
    
    # ==========================================
    # BUSCA ROBUSTA PELO GROUND TRUTH
    # ==========================================
    colunas_dataset = df_test.columns.tolist()
    colunas_possiveis_label = ['Label', 'label', 'Anomaly', 'anomaly', 'Is_Anomaly', 'pred_is_anomaly']
    coluna_alvo = next((col for col in colunas_possiveis_label if col in colunas_dataset), None)

    if coluna_alvo:
        y_verdadeiro = df_test[coluna_alvo].apply(
            lambda x: 1 if str(x).strip().lower() in ['anomaly', '1', 'true', 'anômalo', 'fail'] else 0
        ).values
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ ALERTA: Nenhuma coluna de rótulo (Ground Truth) encontrada.")
        y_verdadeiro = None

    # Aplicação do modelo
    df_resultado, _ = anomaly_detector.process_log_anomalies(
        df_original=df_test, 
        X_tfidf=X_test, 
        y_true=y_verdadeiro,
        model=modelo_treinado 
    )

    # [MARCADOR T1]: Detecção Concluída
    tracker.mark_detected(lote_id)

    # Limpeza final de nulos
    df_resultado = df_resultado.dropna(subset=['Raw_Log'])
    df_resultado['Raw_Log'] = df_resultado['Raw_Log'].astype(str).str.strip()
    df_resultado = df_resultado[df_resultado['Raw_Log'] != ""]

    # ==========================================
    # AGRUPAMENTO TOPOLÓGICO (DBSCAN)
    # ==========================================
    score_silhueta = None
    mask_anomalias = (df_resultado['pred_is_anomaly'] == 1).values
    qtd_anomalias = mask_anomalias.sum()

    if qtd_anomalias > 2:
        print(f"[{time.strftime('%H:%M:%S')}] 🧩 Agrupando anomalias com DBSCAN...")
        
        # Filtra a matriz densa usando máscara booleana (Evita erros de índice NumPy x Pandas)
        X_anomalias_denso = X_test[mask_anomalias]
        
        clusterizador = DBSCAN(eps=0.5, min_samples=4)
        labels_clusters = clusterizador.fit_predict(X_anomalias_denso)
        
        # Adiciona a coluna de cluster no dataframe final
        df_resultado.loc[mask_anomalias, 'cluster_id'] = labels_clusters
        
        if len(set(labels_clusters)) > 1 and len(set(labels_clusters) - {-1}) > 0:
            score_silhueta = silhouette_score(X_anomalias_denso, labels_clusters)
            print(f"[{time.strftime('%H:%M:%S')}] 📐 Silhouette Score do Incidente: {score_silhueta:.4f}")

    # Salva o resultado final processado
    df_resultado.to_parquet(caminho_parquet, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Processamento concluído! Salvo em: {caminho_parquet}")

    # [MARCADOR T2]: Correlação Topológica Concluída
    print(f"[{time.strftime('%H:%M:%S')}] 🕸️ Correlação topológica em Grafos disponível.")
    tracker.mark_isolated(lote_id)
    
    # Consolidação das Métricas
    resultados_metricas = tracker.calculate_results()
    
    # Injeta a Silhouette Score se houver agrupamento válido
    if score_silhueta is not None:
        resultados_metricas["Silhouette_Score"] = round(float(score_silhueta), 4)

    # Adiciona a data e hora do lote para o eixo X do gráfico de histórico
    resultados_metricas["Timestamp_Lote"] = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"[{time.strftime('%H:%M:%S')}] 📊 Métricas do Lote: {resultados_metricas}")
    
    os.makedirs("resultados", exist_ok=True)
    
    # 1. Salva o status atual para os cards superiores
    with open("resultados/metricas_rca.json", "w", encoding="utf-8") as f:
        json.dump(resultados_metricas, f)

    # 2. SISTEMA DE HISTÓRICO (Append-Only)
    historico_path = "resultados/historico_metricas.json"
    historico_dados = []
    
    # Se o histórico já existir, carrega ele primeiro
    if os.path.exists(historico_path):
        try:
            with open(historico_path, "r", encoding="utf-8") as f:
                historico_dados = json.load(f)
        except json.JSONDecodeError:
            pass # Se estiver corrompido, começa um novo
            
    # Adiciona o resultado do lote atual e salva
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
    """Garante que o dashboard atual feche se o main.py "crashar" ou fechar."""
    global processo_dashboard
    if processo_dashboard is not None:
        processo_dashboard.terminate()
        processo_dashboard.wait()

atexit.register(fechar_dashboard_atual)

if __name__ == "__main__":
    print("🧹 Limpando processos fantasmas antigos...")
    limpar_processos_antigos()
    time.sleep(1) 
    
    print("🚀 Iniciando o Sistema de Detecção de Anomalias...")
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
            print("⏳ Aguardando 200s (ou até o usuário mudar alguma configuração na tela)...\n")
            
            if os.path.exists("config.json"):
                ultimo_json_modificado = os.path.getmtime("config.json")
            
            for _ in range(20):
                time.sleep(10)
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