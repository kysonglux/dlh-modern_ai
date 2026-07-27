#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
Logistic_Regression_Model = __import__('5-logisitc_regression').Logistic_Regression_Model

seed = 42
np.random.seed(seed)

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2,
    random_state=seed
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

model = Logistic_Regression_Model(random_state=seed)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("################################# Logistic Regression Model #################################\n")
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print("\nDecision Boundary Plot:")
plt.figure(figsize=(18, 5))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis', alpha=0.7)
plt.title("Decision Boundary – Understanding Class Separation in Feature Space", fontsize=14)
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
class0_patch = mpatches.Patch(color=plt.cm.viridis(0.0), label='Class 0')
class1_patch = mpatches.Patch(color=plt.cm.viridis(1.0), label='Class 1')
plt.legend(handles=[class0_patch, class1_patch], title="Classes")
plt.savefig("5_main_1.png")
plt.show()