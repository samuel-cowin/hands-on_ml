# Decision Trees are another method of machine_learning that are capable of performing regression or classification
# Sci-kit Learn provides functionality in order to implement this model
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris["data"][:, 2:]
y = iris["target"]

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
# Generating the decision tree output .dot file
export_graphviz(tree_clf, out_file="iris_tree.dot",
                feature_names=iris["feature_names"][2:], class_names=iris["target_names"], rounded=True, filled=True)

# Predicting the class based on the probabilities from the decision tree
prob = tree_clf.predict_proba([[5, 1.5]])
pred = tree_clf.predict([[5, 1.5]])
print('Decision Tree Prediction of {} with probabilities {}.'.format(pred, prob))

# As mentioned before, regression is also possible with decision trees
X = 2 * np.random.randint(-100, 100, (1000, 1))/100.0
y = 4 + 3 * X + 15 * pow(X, 2) + np.random.randn(1000, 1)
plt.scatter(X, y)
plt.axis([-2, 2, -100, 100])
plt.show()
tree_reg = DecisionTreeRegressor(max_depth=2, min_samples_leaf=10)
tree_reg.fit(X, y)
export_graphviz(tree_reg, out_file="quadratic_tree.dot",
                rounded=True, filled=True)