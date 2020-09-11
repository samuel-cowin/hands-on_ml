from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Split the data into train and test
def train_test_split(train_index, X, y):
    X_train = X[:train_index, :]
    y_train = y[:train_index]
    X_test = X[train_index:, :]
    y_test = y[train_index:]
    return X_train, y_train, X_test, y_test

# Singular Value Decomposition
def svd_components(X, print_=False):
    X_centered = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_centered)
    if print_:
        print('Principal Components are: {}'.format(Vt))
    return Vt, X_centered

# Dimensionality reduction through projection
def n_components_projection(X_centered, Vt, n=2, print_=False):
    W = np.transpose(Vt)[:,:n]
    X_p = np.dot(X_centered, W)
    if print_:
        print('Projection: {}'.format(X_p))
    return X_p

# Number of components for percentage of variance explained
def components_from_variance(X_train, ratio, model, n_components=3, plot=False):
    pca = model
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_, dtype=float)
    if plot:
        plot_cs = np.append([0], cumsum, 0)
        plt.plot(plot_cs)
        plt.axis([0, n_components+1, 0, 1])
        plt.show()
    return np.argmax(cumsum >= ratio) + 1

# Custom scorer for grid search
def kcpa_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)