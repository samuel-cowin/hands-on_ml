from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def cross_validation_sk(X_train, y_train, classifier, nS=3, rS=42):
    skfolds = StratifiedKFold(n_splits=nS, random_state=rS)
    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(classifier)
        X_train_folds, y_train_folds, X_test_fold, y_test_fold = train_test_split(
            train_index, test_index, X_train, y_train)

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct/len(y_pred))


def train_test_split(train_index, test_index, X_train, y_train):
    X_train_folds = X_train[train_index]
    y_train_folds = y_train[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train[test_index]

    return X_train_folds, y_train_folds, X_test_fold, y_test_fold
