#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
build_decision_tree = __import__('0-build').build_decision_tree
get_pruning_path = __import__('6-pruning_path').get_pruning_path
prune_and_evaluate_trees = __import__('7-prune_decision_tree').prune_and_evaluate_trees


wine = load_wine()
X = wine.data
y = wine.target

seed = 4000
min_samples_leaf =2
min_samples_split=3

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

clf = build_decision_tree(min_samples_leaf=min_samples_leaf,
                          min_samples_split=min_samples_split,
                          random_state=seed)

ccp_alphas, _ = get_pruning_path(clf, X_train, y_train)

clfs, train_scores, test_scores = prune_and_evaluate_trees(
    X_train, y_train, X_test, y_test, ccp_alphas, seed,
    min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

print("Training Scores:", train_scores)
print("Testing Scores:", test_scores)

plt.figure(figsize=(8, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='Train Accuracy', drawstyle='steps-post')
plt.plot(ccp_alphas, test_scores, marker='o', label='Test Accuracy', drawstyle='steps-post')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Effect of ccp_alpha on Train and Test Accuracy')
plt.legend()
plt.grid(True)
plt.show()