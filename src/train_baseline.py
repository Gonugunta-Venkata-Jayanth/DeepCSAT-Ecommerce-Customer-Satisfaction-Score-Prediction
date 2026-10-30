# src/train_baseline.py
"""Train the TF-IDF + Logistic Regression baseline used by predict.py / the Streamlit app.

Run from the repository root:
    python src/train_baseline.py

Saves:
    models/tfidf_vectorizer.pkl   (TfidfVectorizer, unigrams + bigrams, 20k features)
    models/lr_model.pkl           (LogisticRegression on the binary target CSAT >= 4)
    models/baseline_metrics.json  (validation and test metrics from this run)
"""
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data_utils import load_data, prepare_dataset

# --- Column names in data/eCommerce_Customer_support_data.csv ---
TEXT_COL = "Customer Remarks"
LABEL_COL = "CSAT Score"
# -----------------------------------------------------------------

DATA_PATH = os.path.join('data', 'eCommerce_Customer_support_data.csv')
MODEL_DIR = 'models'
RANDOM_STATE = 42


def evaluate(name, clf, X, y):
    preds = clf.predict(X)
    metrics = {
        'accuracy': round(float(accuracy_score(y, preds)), 4),
        'macro_f1': round(float(f1_score(y, preds, average='macro')), 4),
        'f1_not_satisfied': round(float(f1_score(y, preds, pos_label=0)), 4),
        'f1_satisfied': round(float(f1_score(y, preds, pos_label=1)), 4),
        'confusion_matrix': confusion_matrix(y, preds).tolist(),
        'n': int(len(y)),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y, preds, target_names=['not satisfied (0)', 'satisfied (1)'], digits=4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:", metrics['confusion_matrix'])
    return metrics


def main():
    print(f"Loading data from: {DATA_PATH}")
    print(f"Using text column: {TEXT_COL}, label column: {LABEL_COL}")
    df = load_data(DATA_PATH, text_cols=[TEXT_COL], label_col=LABEL_COL)

    # Same cleaning, CSAT >= 4 -> 1 binarisation and stratified 75/15/10 split as train_lstm.py
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(df, random_state=RANDOM_STATE)
    print(f"Rows with remarks: {len(df)} | train {len(X_train)} / val {len(X_val)} / test {len(X_test)}")

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_tfidf, y_train)

    val_metrics = evaluate('Validation', clf, tfidf.transform(X_val), y_val)
    test_metrics = evaluate('Test', clf, tfidf.transform(X_test), y_test)
    majority = round(float(max(y_test.mean(), 1 - y_test.mean())), 4)
    print(f"\nMajority-class accuracy on test (always predict the most common class): {majority:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'lr_model.pkl'))
    with open(os.path.join(MODEL_DIR, 'baseline_metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump({'model': 'TF-IDF (1-2 grams, 20k features) + LogisticRegression(max_iter=200)',
                   'target': 'CSAT Score >= 4 -> satisfied (1), else 0',
                   'split': 'stratified 75/15/10, random_state=42',
                   'validation': val_metrics, 'test': test_metrics,
                   'test_majority_class_accuracy': majority}, fh, indent=2)
    print(f"\n✅ Saved tfidf_vectorizer.pkl, lr_model.pkl and baseline_metrics.json to: {MODEL_DIR}")


if __name__ == '__main__':
    main()
