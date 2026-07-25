#!/usr/bin/env python3
"""generates a detailed classfication report"""

from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """generates a detailed classfication report"""
    return metrics.classification_report(true_labels,
                                         predicted_labels,
                                         target_names=class_names)
