import modules.parse_system as parse_system
import modules.dowloand_dataset_hugging as dowloand_dataset_hugging
import modules.anomaly_detector as anomaly_detector
import modules.preprocessor as preprocessor
import modules.visualizer as visualizer
from datetime import date
import pandas as pd

date = date.today().strftime("%Y-%m-%d")

if __name__ == "__main__":
    contamination = 0.02

    # 1. Leitura e Parse
    read_generic = parse_system.read_dir_to_temps("docker/meus_logs")
    df_list = []
    for path in read_generic:
        df_p = parse_system.automatic_parse(path)
        if not df_p.empty:
            df_list.append(df_p)

    if not df_list:
        print("Nenhum dado encontrado.")
        exit()

    # Variável consolidada
    df_logs = pd.concat(df_list, ignore_index=True)

    # 2. Vetorização
    X_tfidf = preprocessor.tfidf_vectorize(df_logs) 
    print(f"Dimensões para Processamento: {X_tfidf.shape}")

    # 3. Detecção de Anomalias (Ajustado os nomes das variáveis)
    df_final = anomaly_detector.process_log_anomalies(df_logs, X_tfidf, contamination)
    
    # 4. Filtragem de Resultados
    anomalias = df_final[df_final['is_anomaly'] == True]
    normais = df_final[df_final['is_anomaly'] == False]

    # 5. Resumo no Console
    print("\n" + "="*30)
    print(f" RESUMO DE ANOMALIAS (Contaminação: {contamination*100}%) ")
    print("="*30)
    print(f"Total de Logs: {len(df_final)}")
    print(f"Anomalias Detectadas: {len(anomalias)}")
    print(f"Logs Normais: {len(normais)}")

    if not anomalias.empty:
        print("\nExemplos de anomalias encontradas:")
        # Exibe colunas originais + o score da IA
        colunas_ver = [c for c in ['Level', 'Event', 'anomaly_score'] if c in anomalias.columns]
        print(anomalias[colunas_ver].head())
    
    print("\n" + "="*30)
    print(" ANÁLISE DE SCORES DE ANOMALIA ")
    print("\nEstatísticas dos Scores de Anomalia:")
    print(anomalias['anomaly_score'].describe())
    print("\n" + "="*30)
    # 6. Salvando para o TCC (Essencial para sua documentação)
    output_name = f"resultado_tcc_{date}.csv"
    df_final.to_csv(output_name, index=False)
    print(f"\nArquivo salvo com sucesso: {output_name}")


    # --- NOVO: GERAÇÃO DE GRÁFICOS ---
    print("\nGerando gráficos para o TCC...")
    
    # 1. Gráfico de Barras (Desbalanceamento)
    visualizer.plot_anomaly_counts(df_final, f"plots/contagem_{date}.png")
    
    # 2. Distribuição de Scores (Explicabilidade)
    visualizer.plot_anomaly_distribution(df_final, f"plots/distribuicao_{date}.png")
    
    # 3. Linha do Tempo (Ouro para o RCA)
    visualizer.plot_anomaly_timeline(df_final, f"plots/timeline_{date}.png")
    
    print("Gráficos salvos na pasta /plots!")


    df_results = pd.read_csv(output_name)
    df_results
    
    graph_result = visualizer.graph_anomaly_network(df_final, output_path=f"anomaly_network_{date}.png")

    graph_result.show()