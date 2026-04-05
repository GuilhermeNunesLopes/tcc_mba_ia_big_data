import modules.parse_system as parse_system
import modules.dowloand_dataset_hugging as dowloand_dataset_hugging
import modules.anomaly_detector as anomaly_detector
import modules.preprocessor as preprocessor
from datetime import date

date = date.today().strftime("%Y-%m-%d")

if __name__ == "__main__":
    #1. Download Dataset
    log_path = dowloand_dataset_hugging.download_and_decompress(
        repo_id="bolu61/loghub_2", 
        filename="data/apache.txt"
    )

    #2. Parsing - Garantir que os DFs tenham as mesmas colunas esperadas pelo vetorizador

    df_apache = parse_system.automatic_parse(log_path)
    df_spark = parse_system.automatic_parse("logpai/Spark/Spark_2k.log")
    # Mostrando os dados do DataFrame para verificar se o parsing foi bem-sucedido
    #print(df_apache.head())
    #print ("----" * 20)
    #print(df_spark.head())

    # 3. Vetorização (Renomeei para refletir que o retorno é um DataFrame, não um objeto)
    # Use o nome do módulo.função se o import for 'import modules.tfidf_vectorize'
    X_apache = preprocessor.tfidf_vectorize(df_apache) 
    X_spark = preprocessor.tfidf_vectorize(df_spark)
    
    print(f"Dimensões Apache: {X_apache.shape}")
    print(f"Dimensões Spark: {X_spark.shape}")

    print("\n" + "="*30)
    print(" BUSCANDO ANOMALIAS NO APACHE ")
    print("="*30)
    
    anomalies_apache = anomaly_detector.detect_anomalies(X_apache, contamination=0.05)
    
    if not anomalies_apache.empty:
        # Exibimos as linhas do df1 original que correspondem aos índices das anomalias
        print(df_apache.loc[anomalies_apache.index, ['Level', 'Event']])
    else:
        print("Nenhuma anomalia detectada no Apache.")

    print("\n" + "="*30)
    print(" BUSCANDO ANOMALIAS NO SPARK ")
    print("="*30)
    
    anomalies_spark = anomaly_detector.detect_anomalies(X_spark, contamination=0.05)
    
    if not anomalies_spark.empty:
        print(df_spark.loc[anomalies_spark.index, ['Level', 'Event']])
    else:
        print("Nenhuma anomalia detectada no Spark.")
    
    normal = anomaly_detector.get_normal_logs(X_apache, contamination=0.05)
    normal_spark = anomaly_detector.get_normal_logs(X_spark, contamination=0.05)

    print("\n" + "="*30)
    print(" MOSTRANDO AS NÃO ANOMALIAS NO APACHE ")
    print("="*30)

    print(df_apache.loc[normal.index, ['Level', 'Event']])
   

    print("\n" + "="*30)
    print(" MOSTRANDO AS NÃO ANOMALIAS NO SPARK ")
    print("="*30)
    
    print(df_spark.loc[normal_spark.index, ['Level', 'Event']])

    print("="*30)
    print(" RESUMO FINAL ")
    print("="*30)
    print(f"Logs normais Apache: {len(normal)}")
    print(f"Logs normais Spark: {len(normal_spark)}")
    print(f"Anomalias Apache: {len(anomalies_apache)}")
    print(f"Anomalias Spark: {len(anomalies_spark)}")