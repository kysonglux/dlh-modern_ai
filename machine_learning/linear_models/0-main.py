#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
Linear_Regression = __import__('0-linear_regression').Linear_Regression

np.random.seed(42)

X1 = np.random.rand(200) * 10
X2 = X1 + np.random.normal(0, 0.05, 200)
X3 = np.random.rand(200) * 5
X = np.column_stack([X1, X2, X3])
y = 4*X1 + 3*X3 + np.random.normal(0, 5, 200)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

lr = Linear_Regression()
lr.fit(X_train, y_train)

print(lr.get_params())

print("\nLinear Regression Coefficients (for [X1, X2, X3]):", lr.coef_)
print("Linear Regression Intercept (bias term):", lr.intercept_)