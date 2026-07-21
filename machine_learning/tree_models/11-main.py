#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
train_tree = __import__('1-train').train_tree
generate_predictions = __import__('3-generate_predictions').generate_predictions
compare_boosting_classifiers = __import__('11-boosting').compare_boosting_classifiers


wine = load_wine()
X = wine.data
y = wine.target

seed = 4000
n_estimators = 20
n_runs = 500

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

model_names = ["adaboost", "gradientboosting", "xgboost", "lightgbm"]

accuracies = {}
avg_train_times = {}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

for name in model_names:

    times = []
    for _ in range(n_runs):
        model = compare_boosting_classifiers(name, n_estimators, seed)
        start = time.time()
        train_tree(model, X_train, y_train)
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / n_runs

    train_tree(model, X_train, y_train)
    y_pred = generate_predictions(model, X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracies[name] = acc
    avg_train_times[name] = avg_time

plt.figure(figsize=(9, 5))
colors = ['blue', 'orange', 'green', 'red']
plt.bar(accuracies.keys(), accuracies.values(), color=colors)
plt.title("Accuracy Comparison of Boosting Classifiers on the Wine Dataset")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(avg_train_times.keys(), avg_train_times.values(), color=colors)
plt.ylabel("Time (seconds)")
plt.title("Average Training Time (500 runs) for Boosting Classifiers")
for i, v in enumerate(avg_train_times.values()):
    plt.text(i, v + 0.001, f"{v:.3f}", ha='center', fontweight='bold')
plt.show()