#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
build_decision_tree = __import__('0-build').build_decision_tree
get_pruning_path = __import__('6-pruning_path').get_pruning_path


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

ccp_alphas, impurities = get_pruning_path(clf, X_train, y_train)
print("CCP alpha values:", ccp_alphas)
print("Total impurities:", impurities)

plt.figure(figsize=(8, 6))
plt.plot(ccp_alphas, impurities, marker='o', drawstyle='steps-post')
plt.xlabel("ccp_alpha")
plt.ylabel("Total Leaf Impurity")
plt.title("Total Leaf Impurity vs ccp_alpha on Training Data")
plt.grid(True)
plt.show()