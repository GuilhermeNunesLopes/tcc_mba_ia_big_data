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
        return PASTAS_DISPONIVEIS, "auto", "iforest", "pca"


def extrair_rotulo_bgl(df_logs):
    """
    O BGL (LogHub) grava o rótulo como o PRIMEIRO token de cada linha bruta:
    "-" para normal, ou um código de alerta (ex.: "APPREAD", "KERNDTLB") para
    anomalia real, seguido do timestamp em epoch — ex.:
        - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ...
        APPREAD 1117869872 2005.06.04 R04-M1-N4-I:J18-U11 ...
    Nem automatic_drain_parse() nem preprocessar_logs_brutos() extraem esse
    rótulo — sem essa coluna 'Label', o motor ao vivo processava TODAS as
    fontes em modo puramente não supervisionado, mesmo quando a fonte era o
    BGL com rótulo real disponível (F1/PR_AUC nunca eram calculados).
    Só ativa a extração se o padrão bater em quase todas as linhas do lote
    (>=90%), para não injetar uma coluna 'Label' incorreta em datasets que
    não seguem o formato do BGL.

    Nota de duplicação: esta é a mesma função de avaliacao_walkforward.py.
    Mantive duplicada aqui (em vez de importar) para não criar acoplamento
    entre o motor ao vivo e o script de avaliação offline — mover para um
    módulo compartilhado (ex.: modules/parse_system.py) é a melhoria #4 do
    Melhorias_MainV4.docx, fora do escopo desta correção pontual.
    """
    if df_logs.empty or 'Raw_Log' not in df_logs.columns:
        return df_logs

    tokens_label = df_logs['Raw_Log'].str.extract(r'^(\S+)\s+\d{10}\s+\d{4}\.\d{2}\.\d{2}\s')[0]
    taxa_casamento = tokens_label.notna().mean()

    if taxa_casamento >= 0.90:
        print(f"🏷️  Rótulo BGL detectado no formato bruto ({taxa_casamento:.1%} das linhas casam o padrão) "
              f"— coluna 'Label' adicionada.")
        df_logs = df_logs.copy()
        df_logs['Label'] = (tokens_label != '-').astype(int)

    return df_logs


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
    df_logs = pd.concat(df_list, ignore_index=True)
    return extrair_rotulo_bgl(df_logs)


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

    # ---- PROCESSAMENTO POR PASTA (não mais concatenado) ----
    # Antes (main_v4/v5): coletar_logs(pastas_ativas) juntava TODAS as pastas
    # configuradas (sistemas sem relação entre si, ex.: BGL de 2005/2006 +
    # Docker/k8s-chaos de agora) num único DataFrame, ordenado
    # cronologicamente e cortado 80/20. Como as fontes não se intercalam no
    # tempo, o corte cronológico acabava caindo inteiro dentro de UMA fonte
    # só (medido: 100% BGL num lote real — as outras 4 pastas configuradas
    # nunca eram avaliadas). O "normal" aprendido pelo Isolation Forest
    # também ficava contaminado por sistemas sem relação entre si. Agora
    # cada pasta tem seu próprio treino/teste/inferência, isolado das
    # demais — e os resultados de todas as fontes são concatenados no MESMO
    # arquivo que o dashboard já espera (resultado_tcc_{today}.parquet),
    # então dashboard.py não precisa mudar.
    resultados_por_fonte = []
    incidentes_do_lote = []
    metricas_por_fonte = []
    silhouettes_por_fonte = []
    contaminacoes_por_fonte = []

    for pasta in pastas_ativas:
        print(f"\n[{time.strftime('%H:%M:%S')}] 📥 '{pasta}': coletando e parseando logs (Drain3)...")
        df_fonte = coletar_logs([pasta])

        if df_fonte.empty:
            print(f"[{time.strftime('%H:%M:%S')}] ⏭️  '{pasta}': nenhum dado válido encontrado. Pulando.")
            continue

        df_fonte = pipeline.preprocessar_logs_brutos(df_fonte)

        if df_fonte.empty:
            print(f"[{time.strftime('%H:%M:%S')}] ⏭️  '{pasta}': todos os logs eram conhecidos (Whitelist). Pulando.")
            continue

        if len(df_fonte) < 20:
            print(f"[{time.strftime('%H:%M:%S')}] ⏭️  '{pasta}': só {len(df_fonte)} logs após pré-processamento "
                  f"(mínimo 20). Pulando este ciclo.")
            continue

        print(f"[{time.strftime('%H:%M:%S')}] 🔀 '{pasta}': dividindo CRONOLOGICAMENTE (80% Passado / 20% Futuro)...")
        df_train, df_test = train_test_split(df_fonte, test_size=0.2, shuffle=False)

        print(f"[{time.strftime('%H:%M:%S')}] 🧠 '{pasta}': treinando {algoritmo_ativo.upper()} com Redução {reducao_ativa.upper()}...")

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
        taxa_contaminacao_usada = resultado["taxa_contaminacao_usada"]
        detalhes_auto_contaminacao = resultado["detalhes_contaminacao_automatica"]

        if detalhes_auto_contaminacao is not None:
            print(f"[{time.strftime('%H:%M:%S')}] 🧮 '{pasta}': contaminação calculada automaticamente = "
                  f"{taxa_contaminacao_usada:.4%} (Modified Z-Score/MAD, "
                  f"threshold={detalhes_auto_contaminacao['z_threshold']}).")
        contaminacoes_por_fonte.append({
            "Fonte": os.path.basename(pasta),
            "Taxa_Contaminacao_Usada": round(float(taxa_contaminacao_usada), 6),
            "Calculo_Automatico": detalhes_auto_contaminacao,
        })

        if y_verdadeiro is None:
            print(f"[{time.strftime('%H:%M:%S')}] ⚠️ '{pasta}': modo Puramente Não Supervisionado (sem rótulo real).")
        elif metricas_ml:
            print(f"[{time.strftime('%H:%M:%S')}] 📊 '{pasta}': F1={metricas_ml.get('F1_Score', 0):.4f} | "
                  f"PR_AUC={metricas_ml.get('PR_AUC', 0):.4f}")
            metricas_por_fonte.append({"Fonte": os.path.basename(pasta), **metricas_ml})

        if score_silhueta is not None:
            print(f"[{time.strftime('%H:%M:%S')}] 📐 '{pasta}': Silhouette Score = {score_silhueta:.4f}")
            silhouettes_por_fonte.append(float(score_silhueta))

        # ---- RASTREAMENTO DE INCIDENTES (MTTD/MTTI) POR CLUSTER REAL ----
        if 'cluster_id' in df_resultado.columns:
            clusters_reais = df_resultado.loc[
                df_resultado['pred_is_anomaly'] == 1, 'cluster_id'
            ].dropna()
            clusters_reais = clusters_reais[clusters_reais != -1].unique()

            for cid in clusters_reais:
                # Chave da fonte pelo CAMINHO COMPLETO, nao pelo nome final:
                # pastas configuradas podem ter o mesmo basename (ex.:
                # 'docker/logs_appficticio' e
                # 'minikube/k8s-chaos/logs_appficticio' -> ambas dao
                # 'logs_appficticio'). Com o basename, incidentes de fontes
                # distintas que recebiam o mesmo numero de cluster colidiam na
                # mesma chave do rastreador: o registro posterior sobrescrevia
                # o anterior, subcontando Total_Incidentes e contaminando
                # t0/t1 (e portanto MTTD/MTTI).
                chave_fonte = pasta.replace(os.sep, "_").replace("/", "_")
                incident_id = f"{lote_id}_{chave_fonte}_cluster{int(cid)}"
                linhas_cluster = df_resultado[df_resultado['cluster_id'] == cid]
                t0_real = linhas_cluster['Timestamp'].min().timestamp()

                # T2 (isolamento) NAO e marcado aqui. A correlacao topologica
                # em grafo, que e o que a Metodologia define como isolamento
                # ("as etapas de agrupamento e correlacao processam a anomalia,
                # relacionam os eventos e indicam sua possivel causa raiz"), so
                # acontece depois que TODAS as fontes do lote foram
                # consolidadas. Marcar t2 aqui, na instrucao seguinte a
                # mark_detected(), fazia t2 - t1 ser da ordem de microssegundos
                # e o MTTI resultar invariavelmente em 0,0 s.
                tracker.start_injection(incident_id, t0=t0_real)
                tracker.mark_detected(incident_id)
                incidentes_do_lote.append(incident_id)

        resultados_por_fonte.append(df_resultado)

    if not resultados_por_fonte:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Nenhuma pasta produziu dados válidos neste lote.")
        if os.path.exists(caminho_parquet):
            os.utime(caminho_parquet, None)
        return

    if not incidentes_do_lote:
        # Mesmo motivo do bloco acima: t2 deste incidente-placeholder passa a
        # ser marcado junto com os demais, apos a correlacao do lote.
        tracker.start_injection(lote_id)
        tracker.mark_detected(lote_id)
        incidentes_do_lote.append(lote_id)

    # Junta os resultados de todas as fontes e limpa nulos
    df_resultado = pd.concat(resultados_por_fonte, ignore_index=True)
    df_resultado = df_resultado.dropna(subset=['Raw_Log'])
    df_resultado['Raw_Log'] = df_resultado['Raw_Log'].astype(str).str.strip()
    df_resultado = df_resultado[df_resultado['Raw_Log'] != ""]

    df_resultado.to_parquet(caminho_parquet, index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] ✅ Processamento concluído! Salvo em: {caminho_parquet} "
          f"({len(resultados_por_fonte)}/{len(pastas_ativas)} fonte(s) processada(s) com sucesso).")

    print(f"[{time.strftime('%H:%M:%S')}] 🕸️ Correlação topológica em Grafos disponível "
          f"({len(incidentes_do_lote) or 1} incidente(s) rastreado(s) neste lote).")

    # Consolidação das Métricas
    # ---- T2 (ISOLAMENTO) DE TODOS OS INCIDENTES DO LOTE ----
    # Marcado neste ponto, e nao dentro do laco por fonte, porque e aqui que
    # a correlacao topologica do lote fica efetivamente disponivel. Assim
    # t2 - t1 passa a medir um intervalo real (deteccao -> correlacao), em vez
    # do zero estrutural que a marcacao consecutiva produzia.
    # Ressalva conhecida: para a primeira fonte processada, esse intervalo
    # inclui o tempo de processamento das fontes seguintes do mesmo lote --
    # propriedade inerente a um pipeline em lote, nao um erro de medicao.
    for _incident_id in incidentes_do_lote:
        tracker.mark_isolated(_incident_id)

    resultados_metricas = tracker.calculate_results()
    tracker.clear_batch()

    # ---- MÉTRICAS AGREGADAS DE TODAS AS FONTES ----
    # Antes: metricas_rca.json guardava UMA pontuação F1/PR_AUC/Silhouette
    # só — a da ÚLTIMA fonte rotulada/com RCA processada no lote; as
    # demais fontes eram calculadas mas descartadas do arquivo final.
    # Agora: as chaves numéricas de metricas_ml (F1_Score, Precision,
    # Recall, PR_AUC etc.) são a MÉDIA entre todas as fontes que tinham
    # rótulo real no lote — mantendo o schema plano que dashboard.py já
    # lê — e "Metricas_Por_Fonte" guarda o valor individual de cada fonte,
    # para citar no TCC.
    if metricas_por_fonte:
        chaves_numericas = sorted({
            chave for m in metricas_por_fonte
            for chave, valor in m.items()
            if chave != "Fonte" and isinstance(valor, (int, float))
        })
        for chave in chaves_numericas:
            valores = [m[chave] for m in metricas_por_fonte if chave in m]
            resultados_metricas[chave] = round(float(np.mean(valores)), 4)
        resultados_metricas["Metricas_Por_Fonte"] = metricas_por_fonte

    if silhouettes_por_fonte:
        resultados_metricas["Silhouette_Score"] = round(float(np.mean(silhouettes_por_fonte)), 4)

    # ---- TAXA DE CONTAMINAÇÃO (ALERTA) — CALCULADA AUTOMATICAMENTE ----
    # Antes: "taxa_contaminacao" vinha do slider "Alerta (%)" do dashboard,
    # digitado manualmente pelo operador (0,1%-10%, padrão 3,0%). Agora,
    # quando config.json traz "taxa_contaminacao": "auto" (novo padrão),
    # pipeline.estimar_contaminacao_automatica() calcula o valor direto da
    # distribuição de anomaly_score de cada fonte (Modified Z-Score/MAD,
    # ver docstring da função). Guardamos aqui a média entre as fontes do
    # lote (chave plana "Taxa_Contaminacao_Media_Calculada", no schema que
    # o dashboard já lê) e o detalhe por fonte, para auditoria/explicação
    # na monografia.
    if contaminacoes_por_fonte:
        resultados_metricas["Taxa_Contaminacao_Media_Calculada"] = round(
            float(np.mean([c["Taxa_Contaminacao_Usada"] for c in contaminacoes_por_fonte])), 6
        )
        resultados_metricas["Contaminacao_Por_Fonte"] = contaminacoes_por_fonte
        resultados_metricas["Metodo_Calculo_Alerta"] = (
            "auto (Modified Z-Score/MAD sobre IsolationForest.score_samples, "
            "Iglewicz & Hoaglin 1993)"
            if taxa_contaminacao_ativa == "auto"
            else "manual (definido pelo operador no dashboard)"
        )

    resultados_metricas["Timestamp_Lote"] = time.strftime('%Y-%m-%d %H:%M:%S')
    resultados_metricas["Tecnica_Reducao"] = reducao_ativa.upper()
    resultados_metricas["Fontes_Processadas"] = [os.path.basename(p) for p in pastas_ativas]

    print(f"[{time.strftime('%H:%M:%S')}] 📊 Métricas do Lote: {resultados_metricas}")

    os.makedirs("resultados", exist_ok=True)

    with open("resultados/metricas_rca.json", "w", encoding="utf-8") as f:
        json.dump(resultados_metricas, f)

    # Cópia arquivada com data e hora (até o minuto) no próprio nome do
    # arquivo: "resultados/metricas_rca.json" é sobrescrito a cada lote, e
    # sem isso não sobra nenhum registro de QUANDO cada execução foi
    # gerada — útil para citar/anexar prints datados na monografia.
    os.makedirs("resultados/historico_execucoes", exist_ok=True)
    timestamp_arquivo = time.strftime('%Y-%m-%d_%H%M')
    caminho_metricas_arquivada = f"resultados/historico_execucoes/metricas_rca_{timestamp_arquivo}.json"
    with open(caminho_metricas_arquivada, "w", encoding="utf-8") as f:
        json.dump(resultados_metricas, f, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] 🗄️ Cópia com data/hora salva em: {caminho_metricas_arquivada}")

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

    print("🚀 Iniciando o Sistema de Detecção de Anomalias (v6)...")
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
