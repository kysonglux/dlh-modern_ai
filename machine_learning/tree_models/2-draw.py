#!/usr/bin/env python3
"""displays the textual structure of a trained decision tree"""


def draw(clf, feature_names, class_names):
    """displays the textual structure of a trained decision tree """
    tree = clf.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value

    def print_node(node_id=0, depth=0):
        """print the tree using recursive function"""
        indent = " " * depth
        if children_left[node_id] != children_right[node_id]:
            feat_name = feature_names[feature[node_id]]
            thresh = threshold[node_id]
            print(f"{indent}if {feat_name} <= {thresh:.2f}:")

            print_node(children_left[node_id], depth + 1)

            print(f"{indent}else:")

            print_node(children_right[node_id], depth + 1)
        else:
            class_idx = value[node_id].argmax()
            print(f"{indent}return {class_names[class_idx]}")

    print_node(0, 0)
