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
        return pd.DataFrame(), None
    # Combine all text data into a single string for each row
    #df['combined'] = df['Date'] + ' ' + df['Level'] + ' ' + df['Source'] + ' ' + df['Event']
    df['Event_Clean'] = df['Event'].apply(clean_log_text)
    df['combined'] = df['Level'] + ' ' + df['Source'] + ' ' + df['Event']
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)  # Limit to top 1000 features
    
    # Fit and transform the combined text data
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    

    # Convert the TF-IDF matrix to a DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=vectorizer.get_feature_names_out(),
        index=df.index
    )
    
    return tfidf_df
