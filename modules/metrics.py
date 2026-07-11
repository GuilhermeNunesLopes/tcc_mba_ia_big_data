import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred):
    """
    Calcula as métricas científicas do modelo.
    y_true: Série com os labels REAIS (True para anomalia, False para normal)
    y_pred: Série com os labels PREDITOS pelo Isolation Forest
    """
    # zero_division=0 evita erros se o modelo não detectar nenhuma anomalia
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return precision, recall, f1, cm