#!/usr/bin/env python3
"""displays the textual structure of a trained decision tree"""

from sklearn import tree


def draw(clf, feature_names, class_names):
    """displays the textual structure of a trained decision tree """
    print(tree.export_text(clf,
                           feature_names=feature_names,
                           class_names=class_names))
