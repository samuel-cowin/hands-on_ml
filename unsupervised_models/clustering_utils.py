import numpy as np

# Split the data into train and test
def train_test_split(train_index, X, y):
    X_train = X[:train_index, :]
    y_train = y[:train_index]
    X_test = X[train_index:, :]
    y_test = y[train_index:]
    return X_train, y_train, X_test, y_test

# Partially labels unsupervised data based on some labels found and a given percentage
def partially_label_data(X_train, y_train, model, n_k=50, percentile_=20):
    X_digits_dist = model.fit_transform(X_train)
    X_cluster_dist = X_digits_dist[np.arange(len(X_train)), model.labels_]
    for i in range(n_k):
        in_cluster = (model.labels_ == i)
        cluster_dist = X_cluster_dist[in_cluster]
        cutoff_distance = np.percentile(cluster_dist, percentile_)
        above_cutoff = (X_cluster_dist > cutoff_distance)
        X_cluster_dist[in_cluster & above_cutoff] = -1
    partially_propagated = (X_cluster_dist != -1)
    X_train_partially_propagated = X_train[partially_propagated]
    y_train_partially_propagated = y_train[partially_propagated]
    return X_train_partially_propagated, y_train_partially_propagated