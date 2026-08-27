"""
main_v7.py — motor de detecção de anomalias, v7.

Este arquivo é uma EVOLUÇÃO ADITIVA de main_v6.py: mesma interface externa
(mesmo config.json, mesmo dashboard em modules/dashboard.py, mesmos
arquivos de saída em resultados/), rodando lado a lado como um ponto de
entrada alternativo — main_v6.py continua existindo e funcionando
exatamente como antes, sem nenhuma dependência do que está aqui.

O QUE MUDA EM RELAÇÃO AO v6 (resumo — detalhes e "porquês" no documento
"Melhorias_MainV7.docx" entregue junto):

  1) Validação cruzada temporal (TimeSeriesSplit) por lote/fonte, em vez
     de um único corte 80/20 — contaminação automática mais estável e,
     opcionalmente, seleção automática de algoritmo (iForest vs. OCSVM)
     pelo F1 cruzado. (modulos_v2/validacao_cruzada.py)
  2) Persistência versionada do modelo treinado por fonte + fallback
     automático para o último modelo bom quando a CV do lote atual sai
     degenerada (dados insuficientes/sem variação/F1 baixo demais).
     (modulos_v2/model_registry.py, modulos_v2/pipeline_v2.py)
  3) Motor "auto-curável": uma falha não tratada em UMA fonte, ou em UM
     ciclo inteiro, não derruba mais o motor inteiro — é registrada e o
     loop continua no próximo ciclo. No v6, uma exceção não tratada em
     processar_logs_em_lote() saía do laço `while True` sem outro `while`
     por fora para retomar — o motor simplesmente parava de vez (o
     `except Exception` do final do arquivo só dorme 60s e encerra o
     processo), deixando o dashboard rodando sozinho, mostrando dados cada
     vez mais desatualizados sem nenhum aviso.
  4) Logging estruturado em arquivo rotativo (logs_motor/motor_v7.log),
     além do console — dá para diagnosticar depois de um problema que
     aconteceu horas atrás. (modulos_v2/logger_setup.py)
  5) Escrita atômica dos arquivos de saída (parquet/JSON) — elimina o
     risco de o dashboard ler um arquivo pela metade se o motor for
     interrompido no meio de uma escrita. (modulos_v2/io_atomico.py)

COMPATIBILIDADE GARANTIDA COM O DASHBOARD ATUAL
------------------------------------------------
modules/dashboard.py não foi tocado. Todo campo que ele já lê
(resultado_tcc_{data}.parquet, metricas_rca.json com
Taxa_Contaminacao_Media_Calculada etc.) continua sendo escrito com o
MESMO nome e MESMO tipo. Os campos novos (relatório de CV, fallback
usado) são adicionados como chaves EXTRAS no JSON — dashboard.py usa
`.get()` para ler essas chaves (conferido em código), então chaves a mais
não quebram nada; só não aparecem na tela até (se algum dia fizer
sentido) alguém decidir adicionar um card novo para elas.

CONFIG.JSON: NENHUMA MUDANÇA OBRIGATÓRIA
------------------------------------------
main_v7.py lê o MESMO config.json do v6 (pastas, taxa_contaminacao,
algoritmo, reducao). Há UMA chave nova e OPCIONAL,
"selecao_automatica_algoritmo" (bool, padrão False) — se ausente (caso do
config.json atual), o comportamento é IDÊNTICO ao v6 nesse aspecto: o
algoritmo treinado é sempre o escolhido em "algoritmo". Ligar essa chave
exigiria também um ajuste pequeno em modules/dashboard.py (um checkbox
novo na tela) para ser controlável pela UI — NÃO fiz esse ajuste aqui
(dashboard.py é um arquivo existente; qualquer edição nele pede
autorização explícita primeiro, como definido no fluxo de trabalho deste
projeto). Por enquanto, dá para ligar manualmente editando o config.json:
    {"pastas": [...], "taxa_contaminacao": "auto", "algoritmo": "iforest",
     "reducao": "pca", "selecao_automatica_algoritmo": true}
"""
import atexit
import json
import os
import platform
import subprocess
import time
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import modules.parse_system as parse_system
import pipeline as pipeline
from modules.config_pastas import PASTAS_DISPONIVEIS
from modules.mttd_mtti import RCA_MetricsTracker
from modulos_v2 import io_atomico, pipeline_v2
from modulos_v2.logger_setup import configurar_logger

logger = configurar_logger()
tracker = RCA_MetricsTracker()

VERSAO_MOTOR = "v7"

# Depois de uma falha não tratada num ciclo inteiro (ver o loop principal),
# espera esse tempo antes de tentar de novo — evita um loop de erro
# consumindo CPU/IO em alta rotação se a causa for persistente (ex.: uma
# pasta configurada ficou inacessível).
SEGUNDOS_ESPERA_APOS_FALHA_DE_CICLO = 30


def ler_configuracoes():
    """Lê o arquivo de configuração gerado pelo Streamlit (mesmo config.json do v6)."""
    try:
        with open("config.json", "r", encoding='utf-8') as f:
            config = json.load(f)
            return (
                config.get("pastas", []),
                config.get("taxa_contaminacao", "auto"),
                config.get("algoritmo", "iforest"),
                config.get("reducao", "pca"),
                bool(config.get("selecao_automatica_algoritmo", False)),
            )
    except (FileNotFoundError, json.JSONDecodeError):
        return PASTAS_DISPONIVEIS, "auto", "iforest", "pca", False


def extrair_rotulo_bgl(df_logs):
    """Idêntica à de main_v6.py — ver lá o porquê da duplicação (docstring original)."""
    if df_logs.empty or 'Raw_Log' not in df_logs.columns:
        return df_logs

    tokens_label = df_logs['Raw_Log'].str.extract(r'^(\S+)\s+\d{10}\s+\d{4}\.\d{2}\.\d{2}\s')[0]
    taxa_casamento = tokens_label.notna().mean()

    if taxa_casamento >= 0.90:
        logger.info(f"🏷️  Rótulo BGL detectado no formato bruto ({taxa_casamento:.1%} das linhas casam o padrão) "
                    f"— coluna 'Label' adicionada.")
        df_logs = df_logs.copy()
        df_logs['Label'] = (tokens_label != '-').astype(int)

    return df_logs


def coletar_logs(pastas_ativas):
    """Idêntica à de main_v6.py."""
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
    logger.info(f"\n[{time.strftime('%H:%M:%S')}] 🔄 Iniciando varredura de logs (motor {VERSAO_MOTOR})...")

    pastas_ativas, taxa_contaminacao_ativa, algoritmo_ativo, reducao_ativa, selecao_auto_algoritmo = ler_configuracoes()

    if not pastas_ativas:
        logger.info(f"[{time.strftime('%H:%M:%S')}] ⏸️ Nenhuma pasta selecionada no painel. Aguardando...")
        return

    lote_id = f"batch_{int(time.time())}"

    os.makedirs("resultados", exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")
    caminho_parquet = f"resultados/resultado_tcc_{today}.parquet"

    resultados_por_fonte = []
    incidentes_do_lote = []
    metricas_por_fonte = []
    silhouettes_por_fonte = []
    contaminacoes_por_fonte = []
    cv_por_fonte = []
    fallback_por_fonte = []

    for pasta in pastas_ativas:
        nome_fonte = os.path.basename(pasta)
        logger.info(f"\n[{time.strftime('%H:%M:%S')}] 📥 '{pasta}': coletando e parseando logs (Drain3)...")

        try:
            df_fonte = coletar_logs([pasta])
        except Exception:
            logger.exception(f"[{time.strftime('%H:%M:%S')}] ❌ '{pasta}': falha ao coletar/parsear logs. Pulando esta fonte neste ciclo.")
            continue

        if df_fonte.empty:
            logger.info(f"[{time.strftime('%H:%M:%S')}] ⏭️  '{pasta}': nenhum dado válido encontrado. Pulando.")
            continue

        df_fonte = pipeline.preprocessar_logs_brutos(df_fonte)

        if df_fonte.empty:
            logger.info(f"[{time.strftime('%H:%M:%S')}] ⏭️  '{pasta}': todos os logs eram conhecidos (Whitelist). Pulando.")
            continue

        if len(df_fonte) < 20:
            logger.info(f"[{time.strftime('%H:%M:%S')}] ⏭️  '{pasta}': só {len(df_fonte)} logs após pré-processamento "
                        f"(mínimo 20). Pulando este ciclo.")
            continue

        logger.info(f"[{time.strftime('%H:%M:%S')}] 🔀 '{pasta}': dividindo CRONOLOGICAMENTE (80% Passado / 20% Futuro)...")
        df_train, df_test = train_test_split(df_fonte, test_size=0.2, shuffle=False)

        logger.info(f"[{time.strftime('%H:%M:%S')}] 🧠 '{pasta}': treinando (Redução {reducao_ativa.upper()}), "
                    f"com validação cruzada temporal...")

        try:
            resultado = pipeline_v2.treinar_e_avaliar_v2(
                df_train, df_test,
                taxa_contaminacao_ativa=taxa_contaminacao_ativa,
                algoritmo_ativo=algoritmo_ativo,
                reducao_ativa=reducao_ativa,
                top_n_termos=5,
                calcular_rca=True,
                nome_fonte=nome_fonte,
                permitir_selecao_automatica_algoritmo=selecao_auto_algoritmo,
            )
        except Exception:
            logger.exception(f"[{time.strftime('%H:%M:%S')}] ❌ '{pasta}': falha no treino/avaliação. Pulando esta fonte neste ciclo.")
            continue

        df_resultado = resultado["df_resultado"]
        metricas_ml = resultado["metricas_ml"]
        y_verdadeiro = resultado["y_verdadeiro"]
        score_silhueta = resultado["score_silhueta"]
        taxa_contaminacao_usada = resultado["taxa_contaminacao_usada"]
        detalhes_auto_contaminacao = resultado["detalhes_contaminacao_automatica"]
        relatorio_cv = resultado["relatorio_validacao_cruzada"]
        algoritmo_usado = resultado["algoritmo_usado"]
        modelo_fallback_usado = resultado["modelo_fallback_usado"]

        if relatorio_cv is not None:
            logger.info(f"[{time.strftime('%H:%M:%S')}] 🧪 '{pasta}': validação cruzada temporal "
                        f"({relatorio_cv['n_splits']} dobras) — "
                        f"contaminação recomendada={relatorio_cv['contaminacao_recomendada']:.4%}"
                        + (f", melhor algoritmo={relatorio_cv['melhor_algoritmo']}" if relatorio_cv['melhor_algoritmo'] else "")
                        + (f" ⚠️ DEGENERADA: {relatorio_cv['motivo_degeneracao']}" if relatorio_cv['degenerada'] else ""))
        else:
            logger.info(f"[{time.strftime('%H:%M:%S')}] ℹ️ '{pasta}': lote pequeno demais para validação cruzada "
                        f"— usando estimativa de contaminação de uma única janela (mesmo comportamento do v6).")

        if modelo_fallback_usado:
            logger.warning(f"[{time.strftime('%H:%M:%S')}] 🛟 '{pasta}': usando o ÚLTIMO MODELO BOM persistido "
                            f"em vez de treinar um novo — motivo: {resultado['motivo_fallback']}")

        if detalhes_auto_contaminacao is not None:
            logger.info(f"[{time.strftime('%H:%M:%S')}] 🧮 '{pasta}': contaminação calculada automaticamente = "
                        f"{taxa_contaminacao_usada:.4%} (Modified Z-Score/MAD, "
                        f"threshold={detalhes_auto_contaminacao['z_threshold']}).")

        contaminacoes_por_fonte.append({
            "Fonte": nome_fonte,
            "Taxa_Contaminacao_Usada": round(float(taxa_contaminacao_usada), 6),
            "Calculo_Automatico": detalhes_auto_contaminacao,
        })
        cv_por_fonte.append({"Fonte": nome_fonte, "Relatorio": relatorio_cv})
        fallback_por_fonte.append({
            "Fonte": nome_fonte,
            "Fallback_Usado": modelo_fallback_usado,
            "Algoritmo_Usado": algoritmo_usado,
            "Motivo": resultado.get("motivo_fallback"),
        })

        if y_verdadeiro is None:
            logger.info(f"[{time.strftime('%H:%M:%S')}] ⚠️ '{pasta}': modo Puramente Não Supervisionado (sem rótulo real).")
        elif metricas_ml:
            logger.info(f"[{time.strftime('%H:%M:%S')}] 📊 '{pasta}': F1={metricas_ml.get('F1_Score', 0):.4f} | "
                        f"PR_AUC={metricas_ml.get('PR_AUC', 0):.4f}")
            metricas_por_fonte.append({"Fonte": nome_fonte, **metricas_ml})

        if score_silhueta is not None:
            logger.info(f"[{time.strftime('%H:%M:%S')}] 📐 '{pasta}': Silhouette Score = {score_silhueta:.4f}")
            silhouettes_por_fonte.append(float(score_silhueta))

        if 'cluster_id' in df_resultado.columns:
            clusters_reais = df_resultado.loc[
                df_resultado['pred_is_anomaly'] == 1, 'cluster_id'
            ].dropna()
            clusters_reais = clusters_reais[clusters_reais != -1].unique()

            for cid in clusters_reais:
                incident_id = f"{lote_id}_{nome_fonte}_cluster{int(cid)}"
                linhas_cluster = df_resultado[df_resultado['cluster_id'] == cid]
                t0_real = linhas_cluster['Timestamp'].min().timestamp()

                tracker.start_injection(incident_id, t0=t0_real)
                tracker.mark_detected(incident_id)
                tracker.mark_isolated(incident_id)
                incidentes_do_lote.append(incident_id)

        resultados_por_fonte.append(df_resultado)

    if not resultados_por_fonte:
        logger.info(f"[{time.strftime('%H:%M:%S')}] ❌ Nenhuma pasta produziu dados válidos neste lote.")
        if os.path.exists(caminho_parquet):
            os.utime(caminho_parquet, None)
        return

    if not incidentes_do_lote:
        tracker.start_injection(lote_id)
        tracker.mark_detected(lote_id)
        tracker.mark_isolated(lote_id)

    df_resultado = pd.concat(resultados_por_fonte, ignore_index=True)
    df_resultado = df_resultado.dropna(subset=['Raw_Log'])
    df_resultado['Raw_Log'] = df_resultado['Raw_Log'].astype(str).str.strip()
    df_resultado = df_resultado[df_resultado['Raw_Log'] != ""]

    # Escrita atômica: ver modulos_v2/io_atomico.py — elimina o risco do
    # dashboard ler um parquet pela metade se o motor for interrompido aqui.
    io_atomico.escrever_parquet_atomico(df_resultado, caminho_parquet)
    logger.info(f"\n[{time.strftime('%H:%M:%S')}] ✅ Processamento concluído! Salvo em: {caminho_parquet} "
                f"({len(resultados_por_fonte)}/{len(pastas_ativas)} fonte(s) processada(s) com sucesso).")

    logger.info(f"[{time.strftime('%H:%M:%S')}] 🕸️ Correlação topológica em Grafos disponível "
                f"({len(incidentes_do_lote) or 1} incidente(s) rastreado(s) neste lote).")

    resultados_metricas = tracker.calculate_results()
    tracker.clear_batch()

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

    if contaminacoes_por_fonte:
        resultados_metricas["Taxa_Contaminacao_Media_Calculada"] = round(
            float(np.mean([c["Taxa_Contaminacao_Usada"] for c in contaminacoes_por_fonte])), 6
        )
        resultados_metricas["Contaminacao_Por_Fonte"] = contaminacoes_por_fonte
        resultados_metricas["Metodo_Calculo_Alerta"] = (
            "auto (Modified Z-Score/MAD sobre IsolationForest.score_samples, com validação cruzada "
            "temporal quando o lote é grande o bastante — Iglewicz & Hoaglin 1993)"
            if taxa_contaminacao_ativa == "auto"
            else "manual (definido pelo operador no dashboard)"
        )

    # ---- CAMPOS NOVOS DO v7 (chaves extras — dashboard.py ignora com segurança) ----
    resultados_metricas["Motor_Versao"] = VERSAO_MOTOR
    resultados_metricas["Selecao_Automatica_Algoritmo_Ativa"] = selecao_auto_algoritmo
    resultados_metricas["Validacao_Cruzada_Por_Fonte"] = cv_por_fonte
    resultados_metricas["Fallback_Modelo_Por_Fonte"] = fallback_por_fonte

    resultados_metricas["Timestamp_Lote"] = time.strftime('%Y-%m-%d %H:%M:%S')
    resultados_metricas["Tecnica_Reducao"] = reducao_ativa.upper()
    resultados_metricas["Fontes_Processadas"] = [os.path.basename(p) for p in pastas_ativas]

    logger.info(f"[{time.strftime('%H:%M:%S')}] 📊 Métricas do Lote: {resultados_metricas}")

    os.makedirs("resultados", exist_ok=True)
    io_atomico.escrever_json_atomico(resultados_metricas, "resultados/metricas_rca.json")

    os.makedirs("resultados/historico_execucoes", exist_ok=True)
    timestamp_arquivo = time.strftime('%Y-%m-%d_%H%M')
    caminho_metricas_arquivada = f"resultados/historico_execucoes/metricas_rca_{timestamp_arquivo}.json"
    io_atomico.escrever_json_atomico(resultados_metricas, caminho_metricas_arquivada, indent=2, ensure_ascii=False)
    logger.info(f"[{time.strftime('%H:%M:%S')}] 🗄️ Cópia com data/hora salva em: {caminho_metricas_arquivada}")

    historico_path = "resultados/historico_metricas.json"
    historico_dados = []
    if os.path.exists(historico_path):
        try:
            with open(historico_path, "r", encoding="utf-8") as f:
                historico_dados = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"[{time.strftime('%H:%M:%S')}] ⚠️ {historico_path} estava corrompido — recomeçando o histórico do zero.")
            historico_dados = []

    historico_dados.append(resultados_metricas)
    io_atomico.escrever_json_atomico(historico_dados, historico_path)


processo_dashboard = None


def limpar_processos_antigos():
    sistema = platform.system()
    try:
        if sistema == "Windows":
            os.system("taskkill /F /IM streamlit.exe >nul 2>&1")
        else:
            os.system("pkill -f 'streamlit' >/dev/null 2>&1")
    except Exception:
        pass


def fechar_dashboard_atual():
    global processo_dashboard
    if processo_dashboard is not None:
        processo_dashboard.terminate()
        processo_dashboard.wait()


atexit.register(fechar_dashboard_atual)

if __name__ == "__main__":
    logger.info("🧹 Limpando processos fantasmas antigos...")
    limpar_processos_antigos()
    time.sleep(1)

    logger.info(f"🚀 Iniciando o Sistema de Detecção de Anomalias ({VERSAO_MOTOR})...")
    logger.info("🖥️ Abrindo o Dashboard no navegador (Sempre na porta 8501)...")

    comando = ["streamlit", "run", "modules/dashboard.py", "--server.port", "8501"]
    processo_dashboard = subprocess.Popen(comando)

    time.sleep(3)

    logger.info("\n⚙️ Iniciando Motor de Processamento de Logs (Background)...")
    logger.info("⚠️ Pressione CTRL+C no terminal para encerrar o motor e o painel.")
    logger.info("-" * 50)

    try:
        ultimo_json_modificado = 0

        while True:
            # ---- MOTOR AUTO-CURÁVEL ----
            # No v6, uma exceção aqui saía do `while True` sem nada por
            # fora para retomá-lo, e o processo terminava de vez (deixando
            # o dashboard órfão, mostrando dados cada vez mais velhos).
            # Aqui, qualquer falha não tratada num ciclo inteiro é
            # registrada (com stack trace completo no log em disco) e o
            # motor segue para o próximo ciclo, em vez de morrer.
            try:
                processar_logs_em_lote()
            except Exception:
                logger.exception(f"[{time.strftime('%H:%M:%S')}] 🔥 Falha não tratada neste ciclo do motor — "
                                  f"o motor CONTINUA rodando (detalhes acima e em logs_motor/motor_v7.log). "
                                  f"Tentando de novo em {SEGUNDOS_ESPERA_APOS_FALHA_DE_CICLO}s...")
                time.sleep(SEGUNDOS_ESPERA_APOS_FALHA_DE_CICLO)

            logger.info("⏳ Aguardando novos lotes ou alteração de filtro na tela...\n")

            if os.path.exists("config.json"):
                ultimo_json_modificado = os.path.getmtime("config.json")

            for _ in range(200):
                time.sleep(1)
                if os.path.exists("config.json"):
                    modificacao_atual = os.path.getmtime("config.json")
                    if modificacao_atual > ultimo_json_modificado:
                        logger.info("\n🔔 Nova configuração detectada! Acordando o motor imediatamente...")
                        break

    except KeyboardInterrupt:
        logger.info("\n🛑 Encerrando o sistema a pedido do usuário...")
        logger.info("✅ Motor e Dashboard encerrados com sucesso.")
