from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD, PCA  # <--- Adicionado PCA
import numpy as np

def clean_log_text(text):
    text = str(text)

    # ==========================
    # Identificadores únicos
    # ==========================

    # UUID
    text = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b',
        'TAG_UUID',
        text
    )

    # Hexadecimal
    text = re.sub(
        r'0x[0-9a-fA-F]+',
        'TAG_HEX',
        text
    )

    # IPv4
    text = re.sub(
        r'\b\d{1,3}(?:\.\d{1,3}){3}\b',
        'TAG_IP',
        text
    )

    # URL
    text = re.sub(
        r'https?://[^\s]+',
        'TAG_URL',
        text
    )

    # Caminhos
    text = re.sub(
        r'/[A-Za-z0-9_.\-/]+',
        'TAG_PATH',
        text
    )

    # ==========================
    # Preserva informações importantes
    # ==========================

    # HTTP Status Code
    text = re.sub(
        r'\b([1-5]\d{2})\b',
        r'HTTP_\1',
        text
    )

    # Oracle
    text = re.sub(
        r'ORA-\d+',
        'TAG_ORACLE_ERROR',
        text,
        flags=re.IGNORECASE
    )

    # SQLSTATE
    text = re.sub(
        r'SQLSTATE\s+[A-Z0-9]+',
        'TAG_SQLSTATE',
        text,
        flags=re.IGNORECASE
    )

    # Java Exceptions
    text = re.sub(
        r'\b[A-Za-z0-9_]*Exception\b',
        'TAG_EXCEPTION',
        text
    )

    # Erros Unix/Linux (errno)
    text = re.sub(
        r'errno=\d+',
        'TAG_ERRNO',
        text,
        flags=re.IGNORECASE
    )

    # ==========================
    # Mascara números restantes
    # ==========================

    text = re.sub(
        r'\b\d+\b',
        'TAG_NUM',
        text
    )

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
    df_clean["combined"] = (
        "LEVEL_" + df_clean["Level"].astype(str) +
        " SOURCE_" + df_clean["Source"].astype(str) +
        " EVENT_" + df_clean["Event_Clean"].astype(str)
    )

    # 4. Treino ou Teste do Vectorizer
    if vectorizer is None:
        # Modo Treino: Cria e ajusta aos dados
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),  # Apenas unigramas
            #ngram_range=(1, 2),  # Unigramas e bigramas
            min_df=1,           
            max_df=0.95,        
            binary=True,        
            use_idf=True,      
            norm='l2'           
        )
        tfidf_matrix = vectorizer.fit_transform(df_clean['combined'])

        density = (tfidf_matrix.nnz /(tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))

        print(f"Densidade TF-IDF: {density:.4%}")
    else:
        # Modo Teste/Inferência: Apenas aplica o vocabulário já aprendido
        tfidf_matrix = vectorizer.transform(df_clean['combined'])

    print("TF-IDF Matrix shape:", tfidf_matrix.shape)
    return tfidf_matrix, vectorizer

def apply_truncated_svd(tfidf_matrix, svd_model=None, n_components=100):
    """
    Reduz a dimensionalidade da matriz esparsa usando TruncatedSVD (LSA).
    """
    if tfidf_matrix.shape[0] == 0:
        return np.array([]), svd_model

    if svd_model is None:
        n_components = min(n_components, tfidf_matrix.shape[1] - 1)
        
        svd_model = TruncatedSVD(n_components=n_components, random_state=43)
        X_reduced = svd_model.fit_transform(tfidf_matrix)
        
        variancia_explicada = svd_model.explained_variance_ratio_.sum() * 100
        print(f"SVD Treinado! {n_components} componentes explicam {variancia_explicada:.2f}% da variância dos logs.")
        
    else:
        X_reduced = svd_model.transform(tfidf_matrix)

    return X_reduced, svd_model

def apply_pca(tfidf_matrix, pca_model=None, n_components=100):
    """
    Reduz a dimensionalidade convertendo a matriz esparsa TF-IDF em densa para aplicar PCA.
    
    Retorna uma matriz densa (numpy array).
    Se 'pca_model' for None, cria e ajusta o modelo (Treino).
    Se passado, apenas transforma os dados (Teste/Inferência).
    """
    if tfidf_matrix.shape[0] == 0:
        return np.array([]), pca_model

    # Converte a matriz esparsa em densa, pois o PCA do scikit-learn exige
    if sp.issparse(tfidf_matrix):
        X_dense = tfidf_matrix.toarray()
    else:
        X_dense = tfidf_matrix

    if pca_model is None:
        # Ajusta n_components para não exceder o número de amostras ou de colunas
        max_possible = min(X_dense.shape[0] - 1, X_dense.shape[1] - 1)
        n_comp = min(n_components, max_possible) if max_possible > 0 else 1
        
        pca_model = PCA(n_components=n_comp, random_state=43)
        X_reduced = pca_model.fit_transform(X_dense)
        
        variancia_explicada = pca_model.explained_variance_ratio_.sum() * 100
        print(f"PCA Treinado! {n_comp} componentes explicam {variancia_explicada:.2f}% da variância dos logs.")
        
    else:
        X_reduced = pca_model.transform(X_dense)

    return X_reduced, pca_model