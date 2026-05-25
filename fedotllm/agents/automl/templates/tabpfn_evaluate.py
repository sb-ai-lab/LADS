import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

def evaluate_model(model, test_features, test_target):
    y_true = test_target.values.ravel() if hasattr(test_target, 'values') else np.array(test_target).ravel()

    if '{%problem%}' == 'classification':
        y_pred = model.predict(test_features)
        y_prob = model.predict_proba(test_features)[:, 1]
        metrics = {
            'roc_auc': float(roc_auc_score(y_true, y_prob)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'mcc': float(matthews_corrcoef(y_true, y_pred)),
        }
    else:
        y_pred = model.{%predict_method%}(test_features)
        corr, _ = spearmanr(y_true, y_pred)
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'spearman': float(corr),
        }

    print(f"Model metrics: {metrics}")
    return metrics
