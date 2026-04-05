import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np


def detect_anomalies(df, contamination=0.05):
    # Select only numeric features for anomaly detection
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Handle the case where there are no numeric features
    if numeric_df.empty:
        print("No numeric features found for anomaly detection.")
        return pd.DataFrame()  # Return empty DataFrame if no numeric features
    
    # Initialize Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=42)
    
    # Fit the model and predict anomalies
    df['anomaly'] = model.fit_predict(numeric_df)
    
    # Filter out the anomalies (where anomaly == -1)
    anomalies = df[df['anomaly'] == -1]
    
    return anomalies.drop(columns=['anomaly'])

def get_normal_logs(df, contamination=0.05):
    # Seleciona apenas recursos numéricos (matriz TF-IDF)
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    model = IsolationForest(contamination=contamination, random_state=42)
    
    # Treina e prediz
    # 1 = Normal, -1 = Anomalia
    df['normal'] = model.fit_predict(numeric_df)
    
    # Filtra apenas os normais
    normal_logs = df[df['normal'] == 1]
    
    return normal_logs