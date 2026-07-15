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

#Iniciando o Tracking do MTTI E MTTR
tracker = RCA_MetricsTracker()
# ==========================================
# MAPEAMENTO DE PASTAS DE LOGS
# ==========================================
def ler_configuracoes():
    """Lê o arquivo de configuração gerado pelo Streamlit."""
    try:
        with open("config.json", "r", encoding='utf-8') as f:
            config = json.load(f)
            return config.get("pastas", []), config.get("taxa_contaminacao", 0.03)
    except (FileNotFoundError, json.JSONDecodeError):
        # Se o dashboard ainda não gerou o arquivo, usa valores padrão seguros
        pastas_padrao = [
        #    "docker/meus_logs",
        #    "logpai/Apache",  
        #    "logpai/Linux",
        #    "logpai/HDFS",
        #    "logpai/OpenSSH",   
        #   "logpai/Zookeeper",
            "minikube/k8s-chaos/logs"
        ] 
        return pastas_padrao, 0.03

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

    # Cria a pasta resultados se ela não existir para evitar erros
    os.makedirs("resultados", exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")
    caminho_parquet = f"resultados/resultado_tcc_{today}.parquet"

    if not df_list:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Nenhum dado válido encontrado nas pastas.")
        # Dá o "toque" no arquivo antigo para o Streamlit sair da tela de carregamento
        if os.path.exists(caminho_parquet):
            os.utime(caminho_parquet, None) 
        return

    df_logs = pd.concat(df_list, ignore_index=True)

    if 'Template' in df_logs.columns and 'Event' not in df_logs.columns:
        df_logs['Event'] = df_logs['Template']
        df_logs['Source'] = df_logs['Source_Folder'] 
        df_logs['Level'] = "INFO"
        

    matriz_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_logs)

    # ==========================================
    # NOVA LÓGICA DE TREINO E TESTE
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 🔀 Dividindo os dados (80% Treino / 20% Teste)...")
    
    # O train_test_split divide o DataFrame e a Matriz Esparsa mantendo os índices alinhados
    df_train, df_test, X_train_tfidf, X_test_tfidf = train_test_split(
        df_logs, matriz_tfidf, test_size=0.2, random_state=42
    )

    # ==========================================
    # REDUÇÃO DE DIMENSIONALIDADE (SVD)
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 📉 Reduzindo dimensionalidade (TruncatedSVD)...")
    
    # 1. Treina o SVD apenas nos dados de treino (passando svd_model=None)
    X_train, modelo_svd = preprocessor.apply_truncated_svd(X_train_tfidf, svd_model=None, n_components=150)
    
    # 2. Aplica a mesma redução nos dados de teste usando o modelo já treinado
    X_test, _ = preprocessor.apply_truncated_svd(X_test_tfidf, svd_model=modelo_svd)

    # A partir daqui, X_train e X_test já são matrizes densas menores e muito mais 
    # ricas em informação útil para a próxima etapa.

    print(f"[{time.strftime('%H:%M:%S')}] 🧠 Treinando modelo Isolation Forest (Fase 1)...")
    # Passo 1: Chama a função passando model=None. Ele vai fazer o .fit() no X_train denso
    _, modelo_treinado = anomaly_detector.process_log_anomalies(
        df_original=df_train, 
        X_tfidf=X_train, 
        model=None 
    )
    
    # ... O resto do seu código continua exatamente igual a partir daqui ...

    print(f"[{time.strftime('%H:%M:%S')}] 🎯 Aplicando inferência e extraindo métricas (Fase 2)...")
    # Passo 2: Busca a coluna de Ground Truth se existir nos datasets do Logpai.
    # (Ajuste o nome 'Label' ou 'Anomaly' conforme o padrão da coluna no seu dataset)
    # ==========================================
    # BUSCA ROBUSTA PELO GROUND TRUTH (RÓTULO REAL)
    # ==========================================
    print(f"[{time.strftime('%H:%M:%S')}] 🔍 Verificando colunas disponíveis no dataset...")
    colunas_dataset = df_test.columns.tolist()
    print(f"Colunas: {colunas_dataset}")

    # Lista de possíveis nomes para a coluna de anomalia nos datasets do Loghub
    colunas_possiveis_label = ['Label', 'label', 'Anomaly', 'anomaly', 'Is_Anomaly', 'is_anomaly']
    
    # Encontra a primeira coluna da lista que exista no dataframe
    coluna_alvo = next((col for col in colunas_possiveis_label if col in colunas_dataset), None)

    if coluna_alvo:
        print(f"[{time.strftime('%H:%M:%S')}] 🎯 Coluna de rótulo encontrada: '{coluna_alvo}'")
        
        # Garante que os rótulos estejam no formato numérico (0 para normal, 1 para anomalia)
        # O Loghub frequentemente usa strings como "Normal", "Anomaly", "-", etc.
        y_verdadeiro = df_test[coluna_alvo].apply(
            lambda x: 1 if str(x).strip().lower() in ['anomaly', '1', 'true', 'anômalo', 'fail'] else 0
        ).values
    else:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ ALERTA: Nenhuma coluna de rótulo (Ground Truth) foi encontrada.")
        print("-> As métricas de precisão, recall e matriz de confusão serão puladas.")
        y_verdadeiro = None
    # ==========================================

    # Passo 3: Passamos o X_test para prever e o modelo_treinado
    df_resultado, _ = anomaly_detector.process_log_anomalies(
        df_original=df_test, 
        X_tfidf=X_test, 
        y_true=y_verdadeiro,
        model=modelo_treinado 
    )
    # ==========================================


    # 1. Remove qualquer linha onde a coluna principal do log seja NaN ou nula
    df_resultado = df_resultado.dropna(subset=['Raw_Log'])
    
    # 2. Garante que os dados sejam texto e remove espaços em branco nas pontas
    df_resultado['Raw_Log'] = df_resultado['Raw_Log'].astype(str).str.strip()
    
    # 3. Filtra e mantém APENAS as linhas que possuem algum conteúdo
    df_resultado = df_resultado[df_resultado['Raw_Log'] != ""]
    
    # 4. Salva o Parquet limpo e enxuto
    df_resultado.to_parquet(caminho_parquet, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Processamento concluído! Salvo em: {caminho_parquet}")
    
     # [MARCADOR T1]: Detecção Concluída
    # O Isolation Forest terminou de classificar o lote inteiro
    tracker.mark_detected(lote_id)

    print(f"[{time.strftime('%H:%M:%S')}] 🕸️ Iniciando correlação topológica em Grafos...")
    
    tracker.mark_isolated(lote_id)
    # Obtém o dicionário com os resultados
    resultados_metricas = tracker.calculate_results()
    
    # Imprime no terminal
    print(f"[{time.strftime('%H:%M:%S')}] 📊 Métricas do Lote: {resultados_metricas}")
    
    # SALVA PARA O STREAMLIT LER
    # Cria a pasta resultados se ela não existir (por segurança)
    os.makedirs("resultados", exist_ok=True)
    with open("resultados/metricas_rca.json", "w", encoding="utf-8") as f:
        json.dump(resultados_metricas, f)
# Variável global para guardar o processo do dashboard
processo_dashboard = None

def limpar_processos_antigos():
    """Mata qualquer processo fantasma do Streamlit antes de iniciar um novo."""
    sistema = platform.system()
    try:
        if sistema == "Windows":
            # Comando Windows para forçar o fechamento do Streamlit
            os.system("taskkill /F /IM streamlit.exe >nul 2>&1")
        else:
            # Comando Linux/Mac para matar o processo
            os.system("pkill -f 'streamlit' >/dev/null 2>&1")
    except Exception:
        pass

def fechar_dashboard_atual():
    """Garante que o dashboard atual feche se o main.py "crashar" ou fechar."""
    global processo_dashboard
    if processo_dashboard is not None:
        processo_dashboard.terminate()
        processo_dashboard.wait()

# Registra a função de limpeza para rodar automaticamente quando o Python fechar
atexit.register(fechar_dashboard_atual)

if __name__ == "__main__":
    print("🧹 Limpando processos fantasmas antigos...")
    limpar_processos_antigos()
    time.sleep(1) # Dá um tempinho para o sistema operacional limpar a memória
    
    print("🚀 Iniciando o Sistema de Detecção de Anomalias...")
    
    # 1. INICIA O DASHBOARD
    print("🖥️ Abrindo o Dashboard no navegador (Sempre na porta 8501)...")
    
    # Forçamos a porta 8501 para garantir que nunca abra em portas diferentes
    comando = ["streamlit", "run", "modules/dashboard.py", "--server.port", "8501"]
    processo_dashboard = subprocess.Popen(comando)
    
    time.sleep(3) # Tempo para o navegador abrir
    
    print("\n⚙️ Iniciando Motor de Processamento de Logs (Background)...")
    print("⚠️ Pressione CTRL+C no terminal para encerrar o motor e o painel.")
    print("-" * 50)
    
    # 2. LOOP DO MOTOR
    try:
        ultimo_json_modificado = 0
        
        while True:
            # Roda a IA e gera o parquet
            processar_logs_em_lote()
            
            print("⏳ Aguardando 200s (ou até o usuário mudar alguma configuração na tela)...\n")
            
            # Anota o horário que o JSON foi salvo pela última vez
            if os.path.exists("config.json"):
                ultimo_json_modificado = os.path.getmtime("config.json")
            
            # Loop de espera inteligente (20 vezes de 10 segundos = 200s)
            for _ in range(20):
                time.sleep(10)
                
                # Se o arquivo existir, verifica se ele foi alterado agora
                if os.path.exists("config.json"):
                    modificacao_atual = os.path.getmtime("config.json")
                    if modificacao_atual > ultimo_json_modificado:
                        print("\n🔔 Nova configuração detectada! Acordando o motor imediatamente...")
                        break # Quebra a espera de 200s e volta para rodar a IA!
            
    except KeyboardInterrupt:
        print("\n🛑 Encerrando o sistema a pedido do usuário...")
        # O atexit fará o trabalho de fechar o Streamlit automaticamente!
        print("✅ Motor e Dashboard encerrados com sucesso.")
        
    except Exception as e:
        print(f"\n⚠️ Erro inesperado no motor: {e}")
        time.sleep(60)