import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def load_data(path: str, text_cols=None, label_col=None):
    df = pd.read_csv(path)
    if text_cols is None:
        for c in ['message','text','review','content','comment','ticket_text','body']:
            if c in df.columns:
                text_cols = [c]; break
    if label_col is None:
        for c in ['rating','satisfaction','label','score','target']:
            if c in df.columns:
                label_col = c; break
    if text_cols is None or label_col is None:
        print("Columns found:", df.columns.tolist())
        raise ValueError("Could not infer text or label column. Specify text_cols and label_col.")
    df = df.dropna(subset=text_cols+[label_col])
    df['text'] = df[text_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df[label_col] = df[label_col].astype(str)
    return df[['text', label_col]].rename(columns={label_col: 'label'})

def basic_clean(text: str):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

def prepare_dataset(df, test_size=0.15, val_size=0.1, random_state=42):
    df['text'] = df['text'].astype(str).apply(basic_clean)
    try:
        df['label'] = df['label'].astype(float)
        df['label_bin'] = (df['label'] >= 4).astype(int)
        y = df['label_bin']
    except Exception:
        df['label'] = df['label'].astype(str)
        mapping = {k:i for i,k in enumerate(sorted(df['label'].unique()))}
        df['label_enc'] = df['label'].map(mapping)
        y = df['label_enc']
    X_train, X_temp, y_train, y_temp = train_test_split(df['text'], y, test_size=test_size+val_size, random_state=random_state, stratify=y)
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_val, random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test
