#!/usr/bin/env python3
"""calculates key evaluation metrics"""
import sklearn.metrics
import numpy as np

def compute_metrics(predictions):
    """calculates key evaluation metrics"""

    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = logits.argmax(-1)

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")
    recall = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
