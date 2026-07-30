from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
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
        #Diminui o numero de feature afim de reduzir o vocabulário lido, pois logs são muito repetitivos
        #max_features=1000,
        #max_features=300,
        #max_features=500,
        max_features=850,
        #ngram_range=(1, 3),#mudando para pegar unigramas, bigramas e trigramas, visto que logs podem ter palavras repetidas e isso pode gerar mais features
        ngram_range=(1, 2),
        # modificando de para pegar unigramas ao invés de bigramas, visto que rodar issso desse jeito está gerando muitos dados
        #ngram_range=(1, 1),
        #stop_words='english',
        stop_words=None,
        sublinear_tf=True,
        use_idf=False,
        #min_df=2,  # Ignora termos que aparecem em menos de 2 logs
        #min_df=3,  # Ignora termos que aparecem em menos de 3 log
        min_df=1,
        #max_df=0.95,
        max_df=0.95,  # Ignora termos que aparecem em mais de 85% dos logs
        token_pattern=r'(?u)\b[\w.-]+\b',
        strip_accents="unicode", #Adiciona suporte a acentos, visto que logs podem ter palavras com acentos
        binary=True,        # Foco na presença do token, não na frequência
        norm='l2'           # Padroniza vetores longos e curtos matematicamente
        )
        tfidf_matrix = vectorizer.fit_transform(df_clean['combined'])

        density = (tfidf_matrix.nnz /(tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))  # nnz = número de elementos não nulos

        print(f"Densidade TF-IDF: {density:.4%}")
    else:
        # Modo Teste/Inferência: Apenas aplica o vocabulário já aprendido
        tfidf_matrix = vectorizer.transform(df_clean['combined'])

    print("TF-IDF Matrix shape:", tfidf_matrix.shape)
    #print("Feature names (vocabulary) do TF-IDF:")
    #print(vectorizer.get_feature_names_out())
    return tfidf_matrix, vectorizer

def apply_truncated_svd(tfidf_matrix, svd_model=None, n_components=100):
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
        
        svd_model = TruncatedSVD(n_components=n_components, random_state=43)
        X_reduced = svd_model.fit_transform(tfidf_matrix)
        
        # Log útil para o seu TCC: quanta informação os componentes retiveram
        variancia_explicada = svd_model.explained_variance_ratio_.sum() * 100
        print(f"SVD Treinado! {n_components} componentes explicam {variancia_explicada:.2f}% da variância dos logs.")
        
    else:
        # Modo Teste/Inferência: Aplica a transformação já aprendida
        X_reduced = svd_model.transform(tfidf_matrix)

    # O retorno X_reduced é um array denso padrão, pronto para K-Means ou Isolation Forest
    return X_reduced, svd_model