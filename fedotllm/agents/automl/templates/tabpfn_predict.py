import numpy as np

def automl_predict(model, features):
    X = features.to_numpy() if hasattr(features, 'to_numpy') else np.array(features)
    predictions = model.{%predict_method%}(X)
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    print(f"Predictions shape: {predictions.shape}")
    return predictions
