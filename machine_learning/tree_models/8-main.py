#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
build_decision_tree = __import__('0-build').build_decision_tree
generate_predictions = __import__('3-generate_predictions').generate_predictions
evaluate = __import__('4-evaluate').evaluate
get_pruning_path = __import__('6-pruning_path').get_pruning_path
prune_and_evaluate_trees = __import__('7-prune_decision_tree').prune_and_evaluate_trees
get_best_alpha = __import__('8-best_ccp_alpha').get_best_alpha


wine = load_wine()
X = wine.data
y = wine.target

seed = 4000
min_samples_leaf =2
min_samples_split=3
class_names = wine.target_names

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

best_alpha, best_pruned_clf = get_best_alpha(clfs, train_scores, test_scores, ccp_alphas)
print(f"Best alpha: {best_alpha}")

y_pred = generate_predictions(best_pruned_clf, X_test)
classification_report = evaluate(y_test, y_pred, class_names)
print("\n         Post Pruned Decision Tree Classifier - Classification Report\n")
print(classification_report)