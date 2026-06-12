from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class TrainingResult:
    metric_name: str
    metric_value: float
    model_summary: str
    leaderboard: Optional[pd.DataFrame] = None
    extra: dict = field(default_factory=dict)


class AbstractAutoMLBackend(ABC):
    """
    Uniform interface for all AutoML backends.
    Subclass this and register in registry.py to add a new backend.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        target: str,
        task_type: str,       # "binary", "multiclass", "regression"
        metric: str,          # e.g. "roc_auc", "rmse", "f1"
        time_limit: int = 120,
        **kwargs,
    ) -> TrainingResult:
        """Train the AutoML model and return a TrainingResult."""
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions for the given dataframe."""
        ...

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional: return class probabilities. Default raises NotImplementedError."""
        raise NotImplementedError(f"{self.name} does not support predict_proba.")

    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        """Return model comparison table if available."""
        return None

    def explain(self) -> str:
        """Return a human-readable description of what was built."""
        return f"{self.name} model"
