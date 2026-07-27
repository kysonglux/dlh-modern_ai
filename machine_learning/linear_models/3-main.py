#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
Linear_Regression = __import__('0-linear_regression').Linear_Regression
lasso_regression = __import__('3-Lasso_regression').lasso_regression
evaluation_metrics_for_regression = __import__('1-regression_evaluation_metrics').evaluation_metrics_for_regression


np.random.seed(42)

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

y_pred_lr = lr.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

mse_lr, rmse_lr, mae_lr, r2_lr = evaluation_metrics_for_regression(y_test, y_pred_lr)
mse_lasso, rmse_lasso, mae_lasso, r2_lasso = evaluation_metrics_for_regression(y_test, y_pred_lasso)

print(f"Linear Regression R²: {r2_lr:.4f}")
print(f"Lasso Regression R²: {r2_lasso:.4f}")
print("\nTrue relevant features: 5 (defined during data generation)")
print(f"Linear uses {np.sum(lr.coef_ != 0)}/50 features")
print(f"Lasso uses {np.sum(lasso.coef_ != 0)}/50 features")

plt.figure(figsize=(9, 6))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'k--', label='Perfect Prediction')

plt.scatter(y_test, y_pred_lr, alpha=1, c='red',
            label=f'Linear Regression (R²={r2_lr:.3f})')
plt.scatter(y_test, y_pred_lasso, alpha=1, c='green',
            label=f'Lasso Regression (R²={r2_lasso:.3f})')

plt.title('True vs Predicted Values (Lasso vs Linear Regression)', fontsize=14)
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("3_main.png")
plt.show()
