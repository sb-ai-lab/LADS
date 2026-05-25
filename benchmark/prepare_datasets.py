"""Generate benchmark CSV datasets from sklearn toy datasets.

Runs once to populate benchmark/data/{breast_cancer,diabetes}/ with
train.csv, test.csv, and sample_submission.csv — the format LADS expects.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

SEED = 42
BASE = Path(__file__).parent / "data"


def save_dataset(X_train, X_test, y_train, y_test, feature_names, target_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df.insert(0, "id", range(len(train_df)))
    train_df[target_name] = y_train

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df.insert(0, "id", range(len(test_df)))
    # test set has target for local evaluation but agent treats it as holdout
    test_df[target_name] = y_test

    sample_sub = pd.DataFrame({"id": test_df["id"], target_name: 0})

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    sample_sub.to_csv(out_dir / "sample_submission.csv", index=False)
    print(f"Saved to {out_dir}: train={train_df.shape}, test={test_df.shape}")


def prepare_breast_cancer():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
    )
    save_dataset(
        X_train, X_test, y_train, y_test,
        feature_names=data.feature_names,
        target_name="target",
        out_dir=BASE / "breast_cancer",
    )


def prepare_diabetes():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED
    )
    save_dataset(
        X_train, X_test, y_train, y_test,
        feature_names=data.feature_names,
        target_name="target",
        out_dir=BASE / "diabetes",
    )


if __name__ == "__main__":
    prepare_breast_cancer()
    prepare_diabetes()
    print("Done.")
