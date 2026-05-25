import joblib
import numpy as np
from tabpfn import {%model_class%}

def train_model(train_features, train_target):
    y = train_target.values.ravel() if hasattr(train_target, 'values') else np.array(train_target).ravel()
    model = {%model_class%}(n_estimators={%n_estimators%}, device='{%device%}')
    model.fit(train_features, y)
    joblib.dump(model, PIPELINE_PATH)
    return model
