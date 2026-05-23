#Pre-process to transform log to numberic data
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

def clean_log_text(text):
    # Substitui IDs hexadecimais, IPs e números por tokens genéricos
    text = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', text)
    text = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', text)
    text = re.sub(r'\b\d+\b', '<NUM>', text)
    return text.lower()

def tfidf_vectorize(df):
    if df.empty:
        return pd.DataFrame()
    
    # 1. Garantir que não existam NaNs (comum em logs mal parseados)
    df = df.fillna("missing")

    # 2. Aplicar a limpeza que você criou
    df['Event_Clean'] = df['Event'].apply(clean_log_text)
    
    # 3. Combinar as colunas USANDO a versão limpa
    # Note que adicionei o Source e Level porque eles são "âncoras" Fortes
    df['combined'] = df['Level'].astype(str) + ' ' + \
                     df['Source'].astype(str) + ' ' + \
                     df['Event_Clean'].astype(str)

    # 4. Configuração Estratégica do Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2), # <--- O SEGREDO: Pega palavras sozinhas e pares (ex: "connection" e "connection failed")
        stop_words=None     # Em logs, palavras como "at" ou "on" podem ser importantes
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['combined'])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=vectorizer.get_feature_names_out(),
        index=df.index # Mantém o índice original para o seu .loc[] funcionar no main
    )
    
    return tfidf_df