#!/usr/bin/env python3

compute_metrics = __import__('4-compute_metrics').compute_metrics
import numpy as np
from types import SimpleNamespace


logits = np.array([
    [2.0, 1.0, 0.1],
    [0.1, 1.5, 0.4],
    [2.0, 0.2, 3.0],
    [0.5, 1.0, 0.3],
    [0.2, 0.1, 2.0]
])

labels = np.array([0, 1, 2, 0, 1])

pred = SimpleNamespace(predictions=logits, label_ids=labels)

metrics = compute_metrics(pred)
print(metrics)