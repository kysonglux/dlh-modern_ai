#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
build_decision_tree = __import__('0-build').build_decision_tree
train_tree = __import__('1-train').train_tree
generate_predictions = __import__('3-generate_predictions').generate_predictions


wine = load_wine()
X = wine.data
y = wine.target

seed = 4000
min_samples_leaf = 2
min_samples_split = 3

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

clf = build_decision_tree(min_samples_leaf=min_samples_leaf,
                          min_samples_split=min_samples_split,
                          random_state=seed)

train_tree(clf, X_train, y_train)

predictions = generate_predictions(clf, X_test)
print("Prediction vs Ground Truth Labels:\n")
for pred, true in zip(predictions, y_test):
    print(f"Predicted: {pred}   |   Actual: {true}")
