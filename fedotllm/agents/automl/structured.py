from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TabPFNConfig(BaseModel):
    problem: ProblemType = Field(
        ...,
        description="Task type: 'classification' for categorical risk labels, 'regression' for continuous scores",
    )
    metric: str = Field(
        ...,
        description=(
            "Primary evaluation metric. "
            "Classification: 'roc_auc' (default), 'mcc' (for imbalanced data), 'accuracy'. "
            "Regression: 'spearman' (ranking quality), 'rmse', 'r2'."
        ),
    )
    predict_method: Literal["predict", "predict_proba"] = Field(
        ...,
        description=(
            "'predict_proba' for classification when ROC-AUC is needed; "
            "'predict' for regression or accuracy-only classification"
        ),
    )
    n_estimators: int = Field(
        32,
        description="TabPFN ensemble size. Default 32 is sufficient for datasets < 1000 samples.",
    )
    device: Literal["cpu", "cuda"] = Field(
        "cpu",
        description="Compute device. Use 'cuda' only when a GPU is confirmed available.",
    )
