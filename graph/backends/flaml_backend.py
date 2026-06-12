"""
FLAML backend — fast, lightweight AutoML by Microsoft.
Install: pip install flaml
"""
from typing import Optional
import pandas as pd

from .base import AbstractAutoMLBackend, TrainingResult

TASK_MAP = {
    "binary":     "classification",
    "multiclass": "classification",
    "regression": "regression",
}


class FLAMLBackend(AbstractAutoMLBackend):

    name = "flaml"

    def __init__(self):
        try:
            from flaml import AutoML  # noqa: F401
        except ImportError:
            raise ImportError("FLAML is not installed. Run: pip install flaml")
        self._model: Optional[object] = None
        self._target: Optional[str] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        task_type: str = "binary",
        metric: str = "roc_auc",
        time_limit: int = 120,
        **kwargs,
    ) -> TrainingResult:
        from flaml import AutoML

        self._target = target
        flaml_task = TASK_MAP.get(task_type.lower(), "classification")

        X = train_df.drop(columns=[target])
        y = train_df[target]

        self._model = AutoML()
        self._model.fit(
            X,
            y,
            task=flaml_task,
            metric=metric.lower(),
            time_budget=time_limit,
            verbose=0,
        )

        best_score = self._model.best_loss  # FLAML stores loss (lower is better)
        best_estimator = self._model.best_estimator

        return TrainingResult(
            metric_name=metric,
            metric_value=float(1 - best_score) if flaml_task == "classification" else float(best_score),
            model_summary=f"Best estimator: {best_estimator} ({metric}: {1 - best_score:.4f})",
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        X = df.drop(columns=[self._target], errors="ignore")
        return pd.Series(self._model.predict(X))

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        X = df.drop(columns=[self._target], errors="ignore")
        return pd.DataFrame(self._model.predict_proba(X))

    def explain(self) -> str:
        if self._model is None:
            return "FLAML model (not yet trained)"
        return (
            f"FLAML AutoML — best estimator: **{self._model.best_estimator}**, "
            f"best config: {self._model.best_config}"
        )
