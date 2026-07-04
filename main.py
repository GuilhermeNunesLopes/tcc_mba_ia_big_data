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
            "docker/meus_logs",
            "logpai/Apache",  
            "logpai/Linux",
            "logpai/HDFS",
            "logpai/OpenSSH",   
            "logpai/Zookeeper",
            "minikube/k8s-chaos/logs"
        ] 
        return pastas_padrao, 0.03

def processar_logs_em_lote():
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 Iniciando varredura de logs...")
    
    pastas_ativas, taxa_contaminacao_ativa = ler_configuracoes()

    if not pastas_ativas:
        print(f"[{time.strftime('%H:%M:%S')}] ⏸️ Nenhuma pasta selecionada no painel. Aguardando...")
        return
    
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
    caminho_csv = f"resultados/resultado_tcc_{today}.csv"

    if not df_list:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Nenhum dado válido encontrado nas pastas.")
        # Dá o "toque" no arquivo antigo para o Streamlit sair da tela de carregamento
        if os.path.exists(caminho_csv):
            os.utime(caminho_csv, None) 
        return

    df_logs = pd.concat(df_list, ignore_index=True)

    if 'Template' in df_logs.columns and 'Event' not in df_logs.columns:
        df_logs['Event'] = df_logs['Template']
        df_logs['Source'] = df_logs['Source_Folder'] 
        df_logs['Level'] = "INFO" 
        
    X_tfidf, vectorizer = preprocessor.tfidf_vectorize(df_logs) 

    print(f"[{time.strftime('%H:%M:%S')}] 🧠 Rodando modelo de Detecção de Anomalias...")
    df_final, model = anomaly_detector.process_log_anomalies(df_logs, X_tfidf, contamination=taxa_contaminacao_ativa)

    # Salvando no local correto
    df_final.to_csv(caminho_csv, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Processamento concluído! Salvo em: {caminho_csv}")

    
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
            # Roda a IA e gera o CSV
            processar_logs_em_lote()
            
            print("⏳ Aguardando 120s (ou até o usuário mudar alguma configuração na tela)...\n")
            
            # Anota o horário que o JSON foi salvo pela última vez
            if os.path.exists("config.json"):
                ultimo_json_modificado = os.path.getmtime("config.json")
            
            # Loop de espera inteligente (60 vezes de 2 segundos = 120s)
            for _ in range(60):
                time.sleep(2)
                
                # Se o arquivo existir, verifica se ele foi alterado agora
                if os.path.exists("config.json"):
                    modificacao_atual = os.path.getmtime("config.json")
                    if modificacao_atual > ultimo_json_modificado:
                        print("\n🔔 Nova configuração detectada! Acordando o motor imediatamente...")
                        break # Quebra a espera de 120s e volta para rodar a IA!
            
    except KeyboardInterrupt:
        print("\n🛑 Encerrando o sistema a pedido do usuário...")
        # O atexit fará o trabalho de fechar o Streamlit automaticamente!
        print("✅ Motor e Dashboard encerrados com sucesso.")
        
    except Exception as e:
        print(f"\n⚠️ Erro inesperado no motor: {e}")
        time.sleep(60)