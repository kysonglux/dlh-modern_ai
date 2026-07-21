#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
build_decision_tree = __import__('0-build').build_decision_tree
train_tree = __import__('1-train').train_tree


wine = load_wine()
X = wine.data
y = wine.target

seed = 4000
min_samples_leaf = 2
min_samples_split = 3
feature_names = wine.feature_names
class_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

clf = build_decision_tree(min_samples_leaf=min_samples_leaf,
                          min_samples_split=min_samples_split,
                          random_state=seed)

train_tree(clf, X_train, y_train)

plt.figure(figsize=(12, 12))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
plt.title("Decision Tree Classifier Trained on All WINE Dataset Features")
plt.savefig("trained_decision_tree.png")
plt.show()