import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Support Vector Machines
# The support vectors are the data points in which the decision boundary is determined
# Due to poor generalizability and sensitivity to outliers of Hard Margin classification, Soft Margin Classification is often chosen for SVMs

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

