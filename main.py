import parse_system
import dowloand_dataset_hugging
from datetime import date

date = date.today().strftime("%Y-%m-%d")

if __name__ == "__main__":
    # Download the dataset
    file = dowloand_dataset_hugging.download_and_decompress(repo_id="bolu61/loghub_2", filename="data/apache.txt")

    # Parse the log file
    print(file)

    df1 = parse_system.automatic_parse(file)
    df2 = parse_system.automatic_parse("logpai/Spark/Spark_2k.log")
    # Print the parsed DataFrame
    print(df1.head())
    print ("----" * 20)
    print(df2.head())