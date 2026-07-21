#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
random_forest = __import__('9-random_forest').random_forest
train_tree = __import__('1-train').train_tree
feature_importance = __import__('10-feature_importance').feature_importance


wine = load_wine()
X = wine.data
y = wine.target

seed = 4000
n = 20
feature_names = wine.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

rf = random_forest(n_estimators=n, random_state=seed)

train_tree(rf, X_train, y_train)

importances, indices = feature_importance(rf)

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[indices])
plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importances")
plt.tight_layout()
plt.show()