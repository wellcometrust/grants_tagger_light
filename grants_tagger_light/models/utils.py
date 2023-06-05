from scipy import sparse as sp
import numpy as np


def format_predictions(Y_pred_proba, classes, threshold=0.5, probabilities=True):
    """
    Formats predictions to output a list of dictionaries

    Y_pred_proba: sparse array or list of predicted probabilites or class
    (i.e. the output of  `.predict` or `.predict_proba` classifier)
    classes: A list of classes the model is able to predict
    threshold: Float between 0 and 1
    probabilities: Whether Y_pred_proba will contain probabilities or just predictions

    Returns:
        A list of dictionaries for each prediction, e.g.
        [{"tag": "Malaria", "Probability": 0.5}, ...]
    """
    tags = []
    for y_pred_proba in Y_pred_proba:
        if sp.issparse(y_pred_proba):
            y_pred_proba = np.asarray(y_pred_proba.todense()).ravel()
        if probabilities:
            tags_i = {
                tag: prob
                for tag, prob in zip(classes, y_pred_proba)
                if prob >= threshold
            }
        else:
            tags_i = [
                tag for tag, prob in zip(classes, y_pred_proba) if prob >= threshold
            ]
        tags.append(tags_i)

    return tags
