"""
AutoGluon backend — best out-of-the-box accuracy for tabular data.
Install: pip install autogluon.tabular
"""
from typing import Optional
import pandas as pd

from .base import AbstractAutoMLBackend, TrainingResult

TASK_TYPE_MAP = {
    "binary":     "binary",
    "multiclass": "multiclass",
    "regression": "regression",
}

METRIC_MAP = {
    "roc_auc": "roc_auc",
    "auc":     "roc_auc",
    "f1":      "f1",
    "rmse":    "root_mean_squared_error",
    "mse":     "mean_squared_error",
    "r2":      "r2",
    "accuracy": "accuracy",
}


class AutoGluonBackend(AbstractAutoMLBackend):

    name = "autogluon"

    def __init__(self):
        try:
            from autogluon.tabular import TabularPredictor  # noqa: F401
        except ImportError:
            raise ImportError(
                "AutoGluon is not installed. Run: pip install autogluon.tabular"
            )
        self._predictor: Optional[object] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        task_type: str = "binary",
        metric: str = "roc_auc",
        time_limit: int = 120,
        **kwargs,
    ) -> TrainingResult:
        from autogluon.tabular import TabularPredictor

        ag_metric = METRIC_MAP.get(metric.lower(), metric)
        ag_problem = TASK_TYPE_MAP.get(task_type.lower(), task_type)

        self._predictor = TabularPredictor(
            label=target,
            problem_type=ag_problem,
            eval_metric=ag_metric,
        ).fit(
            train_df,
            time_limit=time_limit,
            presets=kwargs.get("presets", "medium_quality"),
        )

        leaderboard = self._predictor.leaderboard(silent=True)
        best_score = leaderboard.iloc[0]["score_val"]
        best_model = leaderboard.iloc[0]["model"]

        return TrainingResult(
            metric_name=ag_metric,
            metric_value=float(best_score),
            model_summary=f"Best model: {best_model} ({ag_metric}: {best_score:.4f})",
            leaderboard=leaderboard,
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._predictor is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        return self._predictor.predict(df)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._predictor is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        return self._predictor.predict_proba(df)

    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        if self._predictor is None:
            return None
        return self._predictor.leaderboard(silent=True)

    def explain(self) -> str:
        if self._predictor is None:
            return "AutoGluon model (not yet trained)"
        lb = self._predictor.leaderboard(silent=True)
        best = lb.iloc[0]
        return (
            f"AutoGluon ensemble — best model: **{best['model']}**, "
            f"validation score: {best['score_val']:.4f}"
        )
