# src/train_lstm.py
import os
import numpy as np
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_utils import load_data, prepare_dataset

# --- MANUAL COLUMN NAMES (for your dataset) ---
TEXT_COL = "Customer Remarks"
LABEL_COL = "CSAT Score"
# ------------------------------------------------

DATA_PATH = os.path.join('data', 'eCommerce_Customer_support_data.csv')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(vocab_size, maxlen, embedding_dim=128):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maxlen),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print(f"Loading data from: {DATA_PATH}")
    print(f"Using text column: {TEXT_COL}, label column: {LABEL_COL}")

    df = load_data(DATA_PATH, text_cols=[TEXT_COL], label_col=LABEL_COL)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(df)
    y_train = y_train.values
    y_val = y_val.values
    y_test = y_test.values

    max_vocab = 20000
    maxlen = 200

    tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxlen)
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=maxlen)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen)

    print("Training Bi-LSTM model...")
    model = build_model(max_vocab, maxlen)

    ckpt_path = os.path.join(MODEL_DIR, 'lstm_model.h5')
    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
    ]

    history = model.fit(
        X_train_seq, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_val_seq, y_val),
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test_seq, y_test)
    print(f"✅ Test accuracy: {acc:.4f}")

    joblib.dump(tokenizer, os.path.join(MODEL_DIR, 'tokenizer.pkl'))
    print(f"✅ Model and tokenizer saved to: {MODEL_DIR}")

if __name__ == '__main__':
    main()
