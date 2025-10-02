# extended_models/models_lstm.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def make_univariate_sequences(series: pd.Series, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    v = series.dropna().astype(float).values
    X, y = [], []
    for i in range(len(v) - seq_len):
        X.append(v[i:i+seq_len]); y.append(v[i+seq_len])
    X = np.asarray(X, dtype=float).reshape(-1, seq_len, 1)
    y = np.asarray(y, dtype=float)
    return X, y

def make_multivariate_sequences(X_df: pd.DataFrame, y: pd.Series, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X_df = X_df.astype(float); y = y.astype(float)
    common = X_df.index.intersection(y.index)
    X_df = X_df.loc[common]; y = y.loc[common]
    X_np, y_np = X_df.values, y.values
    n, n_feat = X_np.shape
    X_seq, y_seq = [], []
    for t in range(seq_len, n):
        X_seq.append(X_np[t-seq_len:t, :]); y_seq.append(y_np[t])
    return np.asarray(X_seq, float), np.asarray(y_seq, float)

def build_lstm_model(input_shape: Tuple[int, int], units: int = 64, dropout: float = 0.0) -> Sequential:
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    if dropout > 0: model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def train_lstm(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.1, patience: int = 5, verbose: int = 1):
    cb = [EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)] if validation_split and patience else []
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=cb, verbose=verbose, shuffle=False)
    return hist

def predict_lstm(model: Sequential, X_seq: np.ndarray) -> np.ndarray:
    return model.predict(X_seq, verbose=0).reshape(-1)
