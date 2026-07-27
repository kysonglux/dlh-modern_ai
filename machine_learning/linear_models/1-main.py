#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
Linear_Regression = __import__('0-linear_regression').Linear_Regression
evaluation_metrics_for_regression = __import__('1-regression_evaluation_metrics').evaluation_metrics_for_regression


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

y_pred = lr.predict(X_test)
mse, rmse, mae, r2 = evaluation_metrics_for_regression(y_test, y_pred)

print("Linear Regression - Model Evaluation on Test Set\n")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
