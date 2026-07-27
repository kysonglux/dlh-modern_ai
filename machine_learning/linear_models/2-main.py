#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
Linear_Regression = __import__('0-linear_regression').Linear_Regression
ridge_regression = __import__('2-ridge_regression').ridge_regression
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

ridge = ridge_regression(random_state=42)
ridge.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_ridge = ridge.predict(X_test)

mse_lr, rmse_lr, mae_lr, r2_lr = evaluation_metrics_for_regression(y_test, y_pred_lr)
mse_ridge, rmse_ridge, mae_ridge, r2_ridge = evaluation_metrics_for_regression(y_test, y_pred_ridge)

print("=== Model Performance with Multicollinearity ===")
print(f"Linear Regression R²: {r2_lr:.4f}")
print(f"Ridge Regression R²: {r2_ridge:.4f}")


plt.figure(figsize=(9, 6))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'k--', label='Perfect Prediction')

plt.scatter(y_test, y_pred_lr, alpha=1, c='red',
            label=f'Linear Regression (R²={r2_lr:.4f})')
plt.scatter(y_test, y_pred_ridge, alpha=1, c='green',
            label=f'Ridge Regression (R²={r2_ridge:.4f})')

plt.title('True vs Predicted Values (Ridge vs Linear Regression)', fontsize=14)
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("2_main.png")
plt.show()