"""
AI for Market Trend Analysis (Technical Indicators + Random Forest)

Task:
Predict the market trend regime 5 trading days ahead.
Trend regime definition (bullish):
    SMA20 > SMA50

Dataset:
- Raw OHLCV CSV (Date, Open, High, Low, Close, Volume)
- Example: spx_daily.csv from Stooq

Outputs:
- Trained model (joblib)
- Evaluation metrics
- Processed feature dataset (CSV)

Run:
    pip install pandas numpy scikit-learn matplotlib joblib
    python market_trend_analysis.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import joblib

RAW_CSV = "spx_daily.csv"  # change if needed
PROCESSED_CSV = "spx_trend_features_2005_2025.csv"
MODEL_OUT = "rf_trend_model.joblib"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()

    # Returns
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    # Volatility (rolling std of daily returns)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # Moving averages
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()

    # Normalized distance of price from moving averages
    for n in [10, 20, 50, 200]:
        df[f"close_sma{n}"] = (df["Close"] - df[f"sma_{n}"]) / df[f"sma_{n}"]

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12-26 EMA) + signal (9 EMA)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volume change (avoid div-by-zero)
    df["vol_chg_5"] = df["Volume"].replace(0, np.nan).pct_change(5)

    return df

def make_dataset(df: pd.DataFrame, horizon_days: int = 5) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["trend_now"] = (df["sma_20"] > df["sma_50"]).astype(int)
    df["target"] = df["trend_now"].shift(-horizon_days)

    features = [
        "ret_1", "ret_5", "ret_10",
        "vol_10", "vol_20",
        "close_sma10", "close_sma20", "close_sma50", "close_sma200",
        "rsi_14",
        "macd", "macd_signal", "macd_hist",
        "vol_chg_5",
    ]
    df = df.dropna().copy()
    return df, features

def evaluate_time_series_cv(X, y) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)
    probs, preds, trues = [], [], []
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    for tr_idx, te_idx in tscv.split(X):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = model.predict_proba(X.iloc[te_idx])[:, 1]
        pr = (p >= 0.5).astype(int)
        probs.append(p)
        preds.append(pr)
        trues.append(y.iloc[te_idx].to_numpy())

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return {
        "ROC_AUC": float(roc_auc_score(trues, probs)),
        "Accuracy": float(accuracy_score(trues, preds)),
        "F1": float(f1_score(trues, preds)),
    }

def main():
    df = pd.read_csv(RAW_CSV)
    df["Date"] = pd.to_datetime(df["Date"])

    # Use a modern period for stability and faster training
    df = df[df["Date"] >= "2005-01-01"].copy()

    df = add_features(df)
    df, features = make_dataset(df, horizon_days=5)

    # Save processed dataset
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"] + features + ["target"]
    df[keep_cols].to_csv(PROCESSED_CSV, index=False)
    print(f"Saved processed dataset: {PROCESSED_CSV} (rows={len(df)})")

    X = df[features].astype(float)
    y = df["target"].astype(int)

    # CV metric (optional)
    cv_metrics = evaluate_time_series_cv(X, y)
    print("TimeSeriesSplit CV metrics:", cv_metrics)

    # Train/test split (chronological)
    split_date = pd.Timestamp("2023-01-01")
    train_mask = df["Date"] < split_date
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    final_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    p = final_model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    metrics = {
        "ROC_AUC": float(roc_auc_score(y_test, p)),
        "Accuracy": float(accuracy_score(y_test, pred)),
        "F1": float(f1_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred)),
        "Recall": float(recall_score(y_test, pred)),
        "ConfusionMatrix": confusion_matrix(y_test, pred).tolist()
    }
    print("Test metrics:", metrics)

    # Save model
    joblib.dump({"model": final_model, "features": features}, MODEL_OUT)
    print(f"Saved model: {MODEL_OUT}")

if __name__ == "__main__":
    main()
