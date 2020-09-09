import numpy as np
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.preprocessing import PolynomialFeatures
from svm_utils import *


# Support Vector Machines
# The support vectors are the data points in which the decision boundary is determined
# Due to poor generalizability and sensitivity to outliers of Hard Margin classification, Soft Margin Classification is often chosen for SVMs

# Classification
# Linear SVM equivalent to using a linear kernal
iris = datasets.load_iris()
X = iris['data'][:, (2,3)]
y = (iris['target'] == 2).astype(np.float64)
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
svm_clf.fit(X,y)
svm_pred = svm_clf.predict([[5.5, 0.7]])
print("Linear SVM Classification: {}\n".format(svm_pred))
# While many datasets are not linearly separable, addition of more features can transform the data into linearly separable 
X, y = make_moons(n_samples=1000, noise=0.15)
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss='hinge')),
])
polynomial_svm_clf.fit(X, y)
svm_pred = polynomial_svm_clf.predict([[1.0, -2.0]])
print("Polynomial SVM Classification: {}\n".format(svm_pred))
# Instead of applying the transformation directly, Kernels can be used and the computational complexity need not be introduced when utilizing sklearn
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5)),
])
poly_kernel_svm_clf.fit(X, y)
svm_pred = poly_kernel_svm_clf.predict([[1.0, -2.0]])
print("Polynomial Kernel SVM Classification: {}\n".format(svm_pred))
# Another alternative is the RBF kernel, which is very useful for irregular shapes and is based on similarity functions
rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001)),
])
rbf_kernel_svm_clf.fit(X, y)
svm_pred = rbf_kernel_svm_clf.predict([[1.0, -2.0]])
print("RBF Kernel SVM Classification: {}\n".format(svm_pred))

# Regression
# In addition to classification, SVMs can be used for regression
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1) 
svm_reg = SVR(kernel="linear", C=100, epsilon=0.1)
svm_reg.fit(X, y.ravel())
X_new = np.array([[0], [2]])
pred = svm_reg.predict(X_new)
print("Linear SVM Regression Prediction: {}\n".format(pred))
# Linear Regression Prediction: [[4.06278739] [9.88949282]]