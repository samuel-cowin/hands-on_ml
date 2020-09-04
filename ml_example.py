from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
from ml_utils import *

"""
Retrieve the data from the image set and split the shuffled training and test data
"""
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

"""
Train SGD for binary classification of 5
"""
check = 0
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
if(check == 1):
    sample = input('Sample from dataset to test\n')
    result = sgd_clf.predict([X[int(sample)]])
    print(result)

"""
Cross-Validation similar to off the shelf methods provided to assess the model through accuracy
"""
# cross_validation_sk(X_train, y_train_5, sgd_clf)

"""
Using sklearn for CV and generating the confusion matrix
"""
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
con_mat = confusion_matrix(y_train_5, y_train_pred)
# Precision - False Positives
print(con_mat[1][1]/(con_mat[0][1]+con_mat[1][1]))
# Recall - False Negatives
print(con_mat[1][1]/(con_mat[1][1]+con_mat[1][0]))
# F1 score combining both metrics
print(f1_score(y_train_5, y_train_pred))
"""
These trade-offs can be further explored through precision_recall_curve in sklearn
And implementing custom thresholds utilizing decision functions
Alternatively, ROC curves with these scores examines the recall/speicificity trade-off 
"""