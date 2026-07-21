#!/usr/bin/env python3

build_decision_tree = __import__('0-build').build_decision_tree


seed = 4000
min_samples_leaf = 2
min_samples_split = 3

clf = build_decision_tree(min_samples_leaf=min_samples_leaf,
                          min_samples_split=min_samples_split,
                          random_state=seed)

print(clf)
print(clf.get_params())
