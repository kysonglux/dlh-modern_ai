#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
build_decision_tree = __import__('0-build').build_decision_tree
train_tree = __import__('1-train').train_tree
generate_predictions = __import__('3-generate_predictions').generate_predictions
evaluate = __import__('4-evaluate').evaluate

wine = load_wine()
X = wine.data
y = wine.target

seed = 4000
min_samples_leaf = 2
min_samples_split = 3
class_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

clf = build_decision_tree(min_samples_leaf=min_samples_leaf,
                          min_samples_split=min_samples_split,
                          random_state=seed)

train_tree(clf, X_train, y_train)

y_pred = generate_predictions(clf, X_test)

classification_report = evaluate(y_test, y_pred, class_names)
print("\n         Decision Tree Classifier - Classification Report\n")
print(classification_report)