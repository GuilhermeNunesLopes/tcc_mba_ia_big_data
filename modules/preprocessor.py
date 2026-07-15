from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
import numpy as np

def clean_log_text(text):
    # Usando prefixos de texto puro para evitar problemas com o tokenizador padrão do TF-IDF
    text = re.sub(r'0x[0-9a-fA-F]+', 'TAG_HEX', str(text))
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', 'TAG_IP', text) # Regex mais preciso para IP
    text = re.sub(r'\b\d+\b', 'TAG_NUM', text)
    return text.lower()

def tfidf_vectorize(df, vectorizer=None):
    """
    Vetoriza os logs. 
    Se 'vectorizer' for passado, apenas transforma (para dados de teste).
    Se 'vectorizer' for None, cria e treina um novo (para dados de treino).
    """
    if df.empty:
        return sp.csr_matrix((0, 0)), vectorizer
    
    # Cria uma cópia para evitar warnings do Pandas (SettingWithCopyWarning)
    df_clean = df.copy()

    # 1. Tratar NaNs
    df_clean = df_clean.fillna("missing")

    # 2. Aplicar a limpeza
    df_clean['Event_Clean'] = df_clean['Event'].apply(clean_log_text)
    
    # 3. Combinar as colunas
    df_clean['combined'] = df_clean['Level'].astype(str) + ' ' + \
                           df_clean['Source'].astype(str) + ' ' + \
                           df_clean['Event_Clean'].astype(str)

    # 4. Treino ou Teste do Vectorizer
    if vectorizer is None:
        # Modo Treino: Cria e ajusta aos dados
        vectorizer = TfidfVectorizer(
        #Diminui o numero de feature afim de reduzir o vocabulário lido, pois logs são muito repetitivos
        #max_features=1000,
        max_features=300,
        # modificando de para pegar unigramas ao invés de bigramas, visto que rodar issso desse jeito está gerando muitos dados
        ngram_range=(1, 2),
        #ngram_range=(1, 1),
            stop_words=None
        )
        tfidf_matrix = vectorizer.fit_transform(df_clean['combined'])
    else:
        # Modo Teste/Inferência: Apenas aplica o vocabulário já aprendido
        tfidf_matrix = vectorizer.transform(df_clean['combined'])


    return tfidf_matrix, vectorizer

def apply_truncated_svd(tfidf_matrix, svd_model=None, n_components=200):
    """
    Reduz a dimensionalidade da matriz esparsa usando TruncatedSVD (LSA).
    
    Retorna uma matriz densa (numpy array), ideal para algoritmos de clusterização.
    Se 'svd_model' for None, cria e ajusta o modelo (Treino).
    Se passado, apenas transforma os dados (Teste/Inferência).
    """
    # Verifica se a matriz está vazia para evitar erros
    if tfidf_matrix.shape[0] == 0:
        return np.array([]), svd_model

    if svd_model is None:
        # Modo Treino: Cria o modelo e ajusta aos dados
        # O número de componentes deve ser menor que o max_features do TF-IDF
        n_components = min(n_components, tfidf_matrix.shape[1] - 1)
        
        svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = svd_model.fit_transform(tfidf_matrix)
        
        # Log útil para o seu TCC: quanta informação os componentes retiveram
        variancia_explicada = svd_model.explained_variance_ratio_.sum() * 100
        print(f"SVD Treinado! {n_components} componentes explicam {variancia_explicada:.2f}% da variância dos logs.")
        
    else:
        # Modo Teste/Inferência: Aplica a transformação já aprendida
        X_reduced = svd_model.transform(tfidf_matrix)

    # O retorno X_reduced é um array denso padrão, pronto para K-Means ou Isolation Forest
    return X_reduced, svd_model