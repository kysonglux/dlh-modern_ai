#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
random_forest = __import__('9-random_forest').random_forest
train_tree = __import__('1-train').train_tree
generate_predictions = __import__('3-generate_predictions').generate_predictions
evaluate = __import__('4-evaluate').evaluate


wine = load_wine()
X = wine.data
y = wine.target

seed = 4000
n = 20
class_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

rf = random_forest(n_estimators=n, random_state=seed)
print(rf)
print(rf.get_params())

train_tree(rf, X_train, y_train)

y_pred = generate_predictions(rf, X_test)

classification_report = evaluate(y_test, y_pred, class_names)
print("\n         Random Forest Classifier - Classification Report\n")
print(classification_report)
