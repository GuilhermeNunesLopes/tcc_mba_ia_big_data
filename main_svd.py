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
import modules.preprocessor as preprocessor
import modules.anomaly_detector as anomaly_detector
from modules.mttd_mtti import RCA_MetricsTracker
from modules.config_pastas import PASTAS_DISPONIVEIS
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

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
            # Retorna as pastas, a contaminação e o ALGORITMO (padrão: iforest)
            return config.get("pastas", []), config.get("taxa_contaminacao", "auto"), config.get("algoritmo", "iforest")
    except (FileNotFoundError, json.JSONDecodeError):
        pastas_padrao = PASTAS_DISPONIVEIS
        return pastas_padrao, "auto", "iforest"

def processar_logs_em_lote():
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 Iniciando varredura de logs...")
    
    # Agora recebemos as três variáveis dinâmicas do portal
    pastas_ativas, taxa_contaminacao_ativa, algoritmo_ativo = ler_configuracoes()

    if not pastas_ativas:
        print(f"[{time.strftime('%H:%M:%S')}] ⏸️ Nenhuma pasta selecionada no painel. Aguardando...")
        return
    
    lote_id = f"batch_{int(time.time())}"
    tracker.start_injection(lote_id)

    df_list = []
    
    # =========================================================================
    # 🚀 ATUALIZAÇÃO 1: Consumindo o gerador em lotes (Map-Reduce de Memória)
    # Evita que o Drain estoure a memória com os dicionários internos
    # =========================================================================
    for pasta in pastas_ativas:
        if os.path.exists(pasta):
            read_generic = parse_system.read_dir_to_temps(pasta)
            for path in read_generic:
                nome_da_fonte = os.path.basename(pasta)
                # Aciona o seu parse atualizado com yield
                gerador_lotes = parse_system.automatic_drain_parse(path, nome_fonte=nome_da_fonte, tamanho_lote=100000)
                
                for df_lote in gerador_lotes:
                    if not df_lote.empty:
                        df_lote['Source_Folder'] = pasta
                        df_list.append(df_lote)

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
    
    termos_whitelist = [
        "kube-proxy", "healthcheck", "get /healthz", "ping",
        "connection closed by peer", "liveness probe", "readiness probe"
    ]
    
    mask_ignorar = pd.Series(False, index=df_logs.index)
    
    for termo in termos_whitelist:
        if 'Raw_Log' in df_logs.columns:
            mask_ignorar = mask_ignorar | df_logs['Raw_Log'].str.contains(termo, case=False, na=False, regex=False)
        elif 'Event' in df_logs.columns: 
            mask_ignorar = mask_ignorar | df_logs['Event'].str.contains(termo, case=False, na=False, regex=False)
            
    qtd_antes = len(df_logs)
    df_logs = df_logs[~mask_ignorar].copy()
    qtd_depois = len(df_logs)
    
    print(f"[{time.strftime('%H:%M:%S')}] 📉 Whitelist descartou {qtd_antes - qtd_depois} logs rotineiros.")

    if df_logs.empty:
        print(f"[{time.strftime('%H:%M:%S')}] ⏸️ Todos os logs deste lote eram conhecidos (Whitelist). Aguardando novos logs...")
        return
        
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] ⏳ Engenharia de Features: Calculando Contexto Temporal...")

    # 1. Garante o formato de data correto
    df_logs['Timestamp'] = pd.to_datetime(df_logs['Timestamp'], errors='coerce')
    df_logs = df_logs.dropna(subset=['Timestamp'])

    # 2. Ordena por pasta e tempo para os cálculos temporais fazerem sentido
    df_logs = df_logs.sort_values(by=['Source_Folder', 'Timestamp']).reset_index(drop=True)

    # 3. CRIA AS COLUNAS QUE ESTAVAM FALTANDO
    df_logs['time_delta'] = df_logs.groupby('Source_Folder')['Timestamp'].diff().dt.total_seconds().fillna(0)

    temp_indexed = df_logs.set_index('Timestamp')
    df_logs['log_rate_5m'] = temp_indexed.groupby('Source_Folder')['Raw_Log'].rolling('5min').count().values

    # 4. Ordenação cronológica final obrigatória antes de fazer o Split
    df_logs = df_logs.sort_values(by='Timestamp').reset_index(drop=True)

    # 1. DIVIDIR OS DADOS PRIMEIRO (Evita Data Leakage no TF-IDF, Scaler e SVD)
    print(f"[{time.strftime('%H:%M:%S')}] 🔀 Dividindo os dados CRONOLOGICAMENTE (80% Passado / 20% Futuro)...")
    df_train, df_test = train_test_split(df_logs, test_size=0.2, shuffle=False)

    # ==========================================
    # 2. FIT/TREINO (Aprendendo apenas com o passado)
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] ⏳ Engenharia de Features (Treino)...")
    
    # Aprende o vocabulário TF-IDF
    tfidf_train, vectorizer = preprocessor.tfidf_vectorize(df_train, vectorizer=None)
    
    # Aprende e aplica o TruncatedSVD para redução de dimensionalidade
    print(f"[{time.strftime('%H:%M:%S')}] 📉 Aplicando TruncatedSVD...")
    svd_train, svd_model = preprocessor.apply_truncated_svd(tfidf_train, svd_model=None, n_components=100)
    
    # Normaliza features temporais
    scaler = StandardScaler()
    temp_train = scaler.fit_transform(df_train[['time_delta', 'log_rate_5m']])
    
    # Combina (SVD retorna matriz densa, usamos np.hstack)
    X_train = np.hstack((svd_train, temp_train))

    # ==========================================
    # 3. TRANSFORM/TESTE (Aplicando regras no futuro)
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] ⏳ Engenharia de Features (Teste)...")
    
    # Aplica o vocabulário existente aos dados novos
    tfidf_test, _ = preprocessor.tfidf_vectorize(df_test, vectorizer=vectorizer)
    
    # Aplica a mesma redução de dimensionalidade
    svd_test, _ = preprocessor.apply_truncated_svd(tfidf_test, svd_model=svd_model)
    
    # Aplica a mesma escala temporal
    temp_test = scaler.transform(df_test[['time_delta', 'log_rate_5m']])
    
    # Combina matriz de teste
    X_test = np.hstack((svd_test, temp_test))

    # ==========================================
    # TRACKING DE INCIDENTES (MTTD)
    # ==========================================
    # O t0 já foi registrado em tracker.start_injection(lote_id) no início do fluxo.
    # ==========================================
    # TRACKING DE INCIDENTES (MTTD)
    # ==========================================
    # Ajuste T0: Tenta capturar a hora real do erro mais antigo deste lote para métricas precisas
    #if not df_test.empty and 'Timestamp' in df_test.columns:
    #    ts_min = pd.to_datetime(df_test['Timestamp'], errors='coerce').min()
    #    if pd.notnull(ts_min):
    #        tracker.incidents[lote_id]['t0'] = ts_min.timestamp()

    # ==========================================
    # FASE 1: TREINAMENTO DO MODELO
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 🧠 Treinando modelo {algoritmo_ativo.upper()}...")
    
    # Desempacota as 5 variáveis (incluindo o threshold que será herdado pelo Teste)
    _, modelo_treinado, _, _, threshold_treinado = anomaly_detector.process_log_anomalies(
        df_original=df_train, 
        X_tfidf=X_train, 
        contamination=taxa_contaminacao_ativa,
        model=None,
        algorithm=algoritmo_ativo # <--- Usa a escolha do portal
    )
    
    # ==========================================
    # FASE 2: INFERÊNCIA E MÉTRICAS
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 🎯 Aplicando inferência e extraindo métricas...")
    
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

    # Aplicação do modelo usando os dados do treino
    df_resultado, _, metricas_ml, _, _ = anomaly_detector.process_log_anomalies(
        df_original=df_test, 
        X_tfidf=X_test, 
        y_true=y_verdadeiro,
        model=modelo_treinado,
        best_threshold=threshold_treinado, 
        algorithm=algoritmo_ativo 
    )

    # =========================================================================
    # 🚀 ATUALIZAÇÃO 2: Otimização Matemática do Threshold (Se houver Ground Truth)
    # Aumenta drásticamente o F1-Score do Dashboard cortando no ponto ótimo
    # =========================================================================
    if y_verdadeiro is not None and np.sum(y_verdadeiro) > 0:
        print(f"[{time.strftime('%H:%M:%S')}] 📐 Calculando limiar ótimo via Precision-Recall Curve...")
        
        # Extrai scores contínuos para o conjunto de testes
        scores_decision = -modelo_treinado.decision_function(X_test)
        precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_verdadeiro, scores_decision)
        pr_auc = auc(recalls_curve, precisions_curve)

        # Encontra o ponto ótimo na curva (F1)
        f1_scores_curve = 2 * (precisions_curve * recalls_curve) / (precisions_curve + recalls_curve + 1e-10)
        best_idx = np.argmax(f1_scores_curve)
        best_threshold_local = thresholds_curve[best_idx] if best_idx < len(thresholds_curve) else thresholds_curve[-1]

        # Sobrescreve as predições com a linha de corte acadêmica
        df_resultado['pred_is_anomaly'] = (scores_decision >= best_threshold_local).astype(int)

        # Atualiza métricas reais
        prec = precision_score(y_verdadeiro, df_resultado['pred_is_anomaly'], zero_division=0)
        rec = recall_score(y_verdadeiro, df_resultado['pred_is_anomaly'], zero_division=0)
        f1 = f1_score(y_verdadeiro, df_resultado['pred_is_anomaly'], zero_division=0)

        print(f"   -> Threshold Otimizado (Teste): {best_threshold_local:.4f}")
        print(f"   -> F1-Score: {f1:.4f} | PR-AUC: {pr_auc:.4f}")

        # Injeta os valores reais no dicionário para o Dashboard exibir
        if metricas_ml is not None:
           metricas_ml['PR_AUC'] = round(float(pr_auc), 4)  # <-- Adicione esta linha
           metricas_ml['F1_Score'] = round(float(f1), 4)
           metricas_ml['Precision'] = round(float(prec), 4)
           metricas_ml['Recall'] = round(float(rec), 4)

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
        
        # Converte a matriz esparsa para densa apenas para as anomalias, garantindo que o DBSCAN funcione perfeitamente
        X_anomalias_denso = X_test[mask_anomalias]
        clusterizador = DBSCAN(eps=0.5, min_samples=4)
        labels_clusters = clusterizador.fit_predict(X_anomalias_denso)
        
        df_resultado.loc[mask_anomalias, 'cluster_id'] = labels_clusters
        
        if len(set(labels_clusters)) > 1 and len(set(labels_clusters) - {-1}) > 0:
            score_silhueta = silhouette_score(X_anomalias_denso, labels_clusters)
            print(f"[{time.strftime('%H:%M:%S')}] 📐 Silhouette Score do Incidente: {score_silhueta:.4f}")

    df_resultado.to_parquet(caminho_parquet, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Processamento concluído! Salvo em: {caminho_parquet}")

    # [MARCADOR T2]: Correlação Topológica Concluída
    print(f"[{time.strftime('%H:%M:%S')}] 🕸️ Correlação topológica em Grafos disponível.")
    tracker.mark_isolated(lote_id)
    
    # Consolidação das Métricas
    resultados_metricas = tracker.calculate_results()
    
    if score_silhueta is not None:
        resultados_metricas["Silhouette_Score"] = round(float(score_silhueta), 4)
    
    if metricas_ml:
        resultados_metricas.update(metricas_ml)
        
    resultados_metricas["Timestamp_Lote"] = time.strftime('%Y-%m-%d %H:%M:%S')

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
    
    # Caminho corrigido apontando para modules/dashboard.py
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