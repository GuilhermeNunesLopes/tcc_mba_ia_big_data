from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse

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
        #ngram_range=(1, 2),
        ngram_range=(1, 1),
            stop_words=None
        )
        tfidf_matrix = vectorizer.fit_transform(df_clean['combined'])
    else:
        # Modo Teste/Inferência: Apenas aplica o vocabulário já aprendido
        tfidf_matrix = vectorizer.transform(df_clean['combined'])

    # 5. Criar DataFrame (Cuidado com memória em datasets gigantes)
    # Se a matriz for muito grande, o ideal é usar a matriz esparsa direto no modelo ML
    # Desativado a matriz densa afim de melhorar o processamento.
    # tfidf_df = pd.DataFrame(
    #    tfidf_matrix.toarray(), 
    #    columns=vectorizer.get_feature_names_out(),
    #    index=df_clean.index
    #)
    
    # Retorna a matriz esparsa E o Vectorizer
    return tfidf_matrix, vectorizer