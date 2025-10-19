import os
import joblib
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from data_utils import basic_clean

MODEL_DIR = 'models'

def load_baseline():
    tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    clf = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
    return tfidf, clf

def load_lstm():
    tokenizer = joblib.load(os.path.join(MODEL_DIR, 'tokenizer.pkl'))
    model = load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    return tokenizer, model

def predict_baseline(texts):
    tfidf, clf = load_baseline()
    X = [basic_clean(t) for t in texts]
    X_tfidf = tfidf.transform(X)
    preds = clf.predict(X_tfidf)
    return preds

def predict_lstm(texts):
    tokenizer, model = load_lstm()
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences([basic_clean(t) for t in texts])
    seqs = pad_sequences(seqs, maxlen=200)
    probs = model.predict(seqs)
    preds = (probs.flatten() >= 0.5).astype(int)
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['baseline','lstm'], default='baseline')
    args = parser.parse_args()
    texts = [args.text]
    if args.model == 'baseline':
        print(predict_baseline(texts))
    else:
        print(predict_lstm(texts))
