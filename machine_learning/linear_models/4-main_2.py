#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
Linear_Regression = __import__('0-linear_regression').Linear_Regression
lasso_regression = __import__('3-Lasso_regression').lasso_regression
get_shap_explainer_and_values = __import__('4-shap').get_shap_explainer_and_values


np.random.seed(42)
feature_names = [f"X{i}" for i in range(1, 51)]

X = np.random.randn(100, 50)
true_coef = np.zeros(50)
true_coef[:5] = [5, 4, 3, 2, 1]
y = X @ true_coef + np.random.normal(0, 3, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

lr = Linear_Regression()
lr.fit(X_train, y_train)

lasso = lasso_regression(random_state=42)
lasso.fit(X_train, y_train)

explainer_lr, shap_values_lr = get_shap_explainer_and_values(lr, X_train, X_test)
explainer_lasso, shap_values_lasso = get_shap_explainer_and_values(lasso, X_train, X_test)

shap_values_lr.feature_names = feature_names
shap_values_lasso.feature_names = feature_names

plt.figure(figsize=(30, 18))  

plt.subplot(2, 2, 1)
shap.plots.bar(shap_values_lr, show=False)
plt.title("SHAP Feature Importance – Linear Regression")
plt.gca().set_xlabel("Mean Absolute SHAP Values Across Samples")

plt.subplot(2, 2, 2)
shap.plots.bar(shap_values_lasso, show=False)
plt.title("SHAP Feature Importance – Lasso Regression")
plt.gca().set_xlabel("Mean Absolute SHAP Values Across Samples")

plt.subplot(2, 2, 3)
shap.plots.beeswarm(shap_values_lr, show=False)
plt.title("SHAP Beeswarm Plot – Linear Regression")

plt.subplot(2, 2, 4)
shap.plots.beeswarm(shap_values_lasso, show=False)
plt.title("SHAP Beeswarm Plot – Lasso Regression")

plt.tight_layout()
plt.subplots_adjust(left=0.2, right=0.8)
plt.savefig("4_main_2.png")
plt.show()