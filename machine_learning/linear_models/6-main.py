#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
Logistic_Regression_Model = __import__('5-logisitc_regression').Logistic_Regression_Model
get_SVM_model = __import__('6-svm').get_SVM_model

seed = 42
np.random.seed(seed)

X, y = make_gaussian_quantiles(
    n_samples=300,
    n_features=2,
    n_classes=3,
    random_state=seed)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=seed)

lr = Logistic_Regression_Model(seed).fit(X_train, y_train)
svm_linear = get_SVM_model("linear",seed).fit(X_train, y_train)
svm_poly = get_SVM_model("poly",seed).fit(X_train, y_train)
svm_rbf = get_SVM_model("rbf",seed).fit(X_train, y_train)

print("Logistic Regression classification report:\n", classification_report(y_test, lr.predict(X_test)))
print("SVM Linear classification report:\n", classification_report(y_test, svm_linear.predict(X_test)))
print("SVM Poly classification report:\n", classification_report(y_test, svm_poly.predict(X_test)))
print("SVM RBF classification report:\n", classification_report(y_test, svm_rbf.predict(X_test)))

print("\nDecision boundary plot:")

def plot_decision_boundary(model, title, subplot):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    subplot.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
    subplot.set_title(title)
    subplot.set_xlabel("Feature 1")
    subplot.set_ylabel("Feature 2")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plot_decision_boundary(lr, "Logistic Regression", axes[0, 0])
plot_decision_boundary(svm_linear, "SVM (Linear Kernel)", axes[0, 1])
plot_decision_boundary(svm_poly, "SVM (Polynomial Kernel)", axes[1, 0])
plot_decision_boundary(svm_rbf, "SVM (RBF Kernel)", axes[1, 1])
plt.tight_layout()
plt.savefig("6_main.png")
plt.show()
