"""TabPFN v2 baseline — pure sklearn API, no LADS orchestration.

Run this to get the raw TabPFN numbers. Compare these metrics against
the LADS+TabPFN pipeline output to confirm the orchestration does not
introduce regression (expected delta < 0.05 on ROC-AUC / Spearman).

Usage:
    python benchmark/tabpfn_baseline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tabpfn import TabPFNClassifier, TabPFNRegressor

SEED = 42
BASE = Path(__file__).parent / "data"


def run_classification(data_dir: Path):
    print("\n=== Breast Cancer (Classification) ===")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    feature_cols = [c for c in train_df.columns if c not in ("id", "target")]
    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df["target"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = TabPFNClassifier(n_estimators=32, device="cpu")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  MCC      : {matthews_corrcoef(y_test, y_pred):.4f}")


def run_regression(data_dir: Path):
    print("\n=== Diabetes (Regression) ===")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    feature_cols = [c for c in train_df.columns if c not in ("id", "target")]
    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["target"].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df["target"].values.astype(float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = TabPFNRegressor(n_estimators=32, device="cpu")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    corr, _ = spearmanr(y_test, y_pred)

    print(f"  Spearman : {corr:.4f}")
    print(f"  RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  R²       : {r2_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    run_classification(BASE / "breast_cancer")
    run_regression(BASE / "diabetes")
    print("\nBaseline complete.")
