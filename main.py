import modules.parse_system as parse_system
import modules.dowloand_dataset_hugging as dowloand_dataset_hugging
import modules.anomaly_detector as anomaly_detector
import modules.preprocessor as preprocessor
from datetime import date
import pandas as pd

date = date.today().strftime("%Y-%m-%d")

if __name__ == "__main__":

    contamination=0.02
    #1. Download Dataset
    log_path = dowloand_dataset_hugging.download_and_decompress(
        repo_id="bolu61/loghub_2", 
        filename="data/apache.txt"
    )

    #2. Parsing - Garantir que os DFs tenham as mesmas colunas esperadas pelo vetorizador

    df_apache = parse_system.automatic_parse(log_path)
    df_spark = parse_system.automatic_parse("logpai/Spark/Spark_2k.log")

    #2.1 Lendo arquivos antes de fazer o parse.
    read_generic = parse_system.read_dir_to_temps("docker/meus_logs")
    
    df_generic = []
    for i in read_generic:
       df_generic_parse = parse_system.automatic_parse(i)

       # É boa prática verificar se o DataFrame não está vazio
       if not df_generic_parse.empty:
          df_generic.append(df_generic_parse)

    if df_generic:
        df_generic_final = pd.concat(df_generic, ignore_index=True)
        print("DataFrame consolidado com sucesso!")
    else:
        df_generic_final = pd.DataFrame() # Cria um DF vazio caso não encontre dados
        print("Nenhum dado encontrado nos arquivos.")
    # Mostrando os dados do DataFrame para verificar se o parsing foi bem-sucedido
    #print(df_apache.head())
    #print ("----" * 20)
    #print(df_spark.head())

    # 3. Vetorização (Renomeei para refletir que o retorno é um DataFrame, não um objeto)
    # Use o nome do módulo.função se o import for 'import modules.tfidf_vectorize'
    X_apache = preprocessor.tfidf_vectorize(df_apache) 
    X_spark = preprocessor.tfidf_vectorize(df_spark)
    X_generic = preprocessor.tfidf_vectorize(df_generic_final)
    
    print(f"Dimensões Apache: {X_apache.shape}")
    print(f"Dimensões Spark: {X_spark.shape}")
    print(f"Dimensões Logs Genéricos: {X_generic.shape}")


    print("\n" + "="*30)
    print(" BUSCANDO ANOMALIAS NO APACHE ")
    print("="*30)
    
    anomalies_apache = anomaly_detector.detect_anomalies(X_apache, contamination=contamination)
    
    if not anomalies_apache.empty:
        # Exibimos as linhas do df1 original que correspondem aos índices das anomalias
        print(df_apache.loc[anomalies_apache.index, ['Level', 'Event']])
    else:
        print("Nenhuma anomalia detectada no Apache.")

    print("\n" + "="*30)
    print(" BUSCANDO ANOMALIAS NO SPARK ")
    print("="*30)
    
    anomalies_spark = anomaly_detector.detect_anomalies(X_spark, contamination=contamination)
    
    if not anomalies_spark.empty:
        print(df_spark.loc[anomalies_spark.index, ['Level', 'Event']])
    else:
        print("Nenhuma anomalia detectada no Spark.")
    

    print("\n" + "="*30)
    print(" BUSCANDO ANOMALIAS NOS LOGS GENÉRICOS ")
    print("="*30)
    
    anomalies_generic = anomaly_detector.detect_anomalies(X_generic, contamination=contamination)
    
    if not anomalies_generic.empty:
        # Exibimos as linhas do df1 original que correspondem aos índices das anomalias
        print(df_generic_final.loc[anomalies_generic.index, ['Level', 'Event']])
    else:
        print("Nenhuma anomalia detectada nos logs genéricos.")


    normal = anomaly_detector.get_normal_logs(X_apache, contamination=contamination)
    normal_spark = anomaly_detector.get_normal_logs(X_spark, contamination=contamination)
    normal_generic = anomaly_detector.get_normal_logs(X_generic, contamination=contamination)


    print("\n" + "="*30)
    print(" MOSTRANDO AS NÃO ANOMALIAS NO APACHE ")
    print("="*30)

    print(df_apache.loc[normal.index, ['Level', 'Event']])
   

    print("\n" + "="*30)
    print(" MOSTRANDO AS NÃO ANOMALIAS NO SPARK ")
    print("="*30)
    
    print(df_spark.loc[normal_spark.index, ['Level', 'Event']])


    print("\n" + "="*30)
    print(" MOSTRANDO AS NÃO ANOMALIAS NOS LOGS GENÉRICOS ")
    print("="*30)
    
    print(df_generic_final.loc[normal_generic.index, ['Level', 'Event']])

    print("="*30)
    print(" RESUMO FINAL ")
    print("="*30)
    print(f"Logs normais Apache: {len(normal)}")
    print(f"Logs normais Spark: {len(normal_spark)}")
    print(f"Logs normais Genéricos: {len(normal_generic)}")
    print(f"Anomalias Apache: {len(anomalies_apache)}")
    print(f"Anomalias Spark: {len(anomalies_spark)}")
    print(f"Anomalias Logs Genéricos: {len(anomalies_generic)}")

    

    anomalies_apache_final = anomalies_apache.sum(axis=1)
    anomalies_spark_final = anomalies_spark.sum(axis=1)
    anomalies_generic_final = anomalies_generic.sum(axis=1)

    # Agora sim, você pode criar o df_final
    df_final = pd.DataFrame({
        "Anomalies_Apache": anomalies_apache_final.reset_index(drop=True),
        "Anomalies_Spark": anomalies_spark_final.reset_index(drop=True),
        "Anomalies_Generic": anomalies_generic_final.reset_index(drop=True)
    })
    
    print(df_final.head())