#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
Standardize = __import__('0-standardize').Standardize

X, _ = load_wine(return_X_y=True)
feature_names = load_wine().feature_names

X_scaled = Standardize(X)

X_df = pd.DataFrame(X, columns=feature_names)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)


sns.set(style="whitegrid")

plt.figure(figsize=(14, 10))
sns.boxplot(data=X_df)
plt.title("Feature Distribution Before Standardization")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
sns.boxplot(data=X_scaled_df)
plt.title("Feature Distribution After Standardization")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.savefig("task_0.png")