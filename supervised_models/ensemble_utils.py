from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Plot classification models
def plot_training_model(X, y, model, l=2):
    plot_decision_regions(X, y, clf=model, legend=l)
    plt.show()

# Split the data into train and test
def train_test_split(train_index, X, y):
    X_train = X[:train_index, :]
    y_train = y[:train_index]
    X_test = X[train_index:, :]
    y_test = y[train_index:]
    return X_train, y_train, X_test, y_test

# Choosens best number of iterations of trees based on minimum MSE
def best_selection_trees(model_fitted, X_train, y_train, X_test, y_test):
    errors = [mean_squared_error(y_test, y_pred) for y_pred in model_fitted.staged_predict(X_test)]
    bst_n_estimators = np.argmin(errors) + 1
    return bst_n_estimators

# Stops iteration of all trees when a minima in MSE is found
def early_stopping_trees(model, X_train, y_train, X_test, y_test, n_estimators=100):
    min_val_error = float("inf")
    error_going_up = 0
    for n_e in range(1, n_estimators):
        model.n_estimators = n_e
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        val_error = mean_squared_error(y_test, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                return model

# For each of the models provided predict accuracy on test data
def clf_accuracy(models, X_train, y_train, X_test, y_test, plot=False):
    for m in models:
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        if plot:
            plot_training_model(np.concatenate((X_train, X_test)),
                                np.concatenate((y_train, y_test)), m, l=2)
        print(m.__class__.__name__, accuracy_score(y_test, pred))