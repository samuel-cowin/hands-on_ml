from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np


# Plot classification models
def plot_training_model(X, y, model, l=2):
    plot_decision_regions(X, y, clf=model, legend=l)
    plt.show()

# Split the data into train and test
def train_test_split_index(train_index, X, y):
    X_train = X[:train_index, :]
    y_train = y[:train_index]
    X_test = X[train_index:, :]
    y_test = y[train_index:]
    return X_train, y_train, X_test, y_test

# Find the closest cluster for the datapoint
def labels_for_data(model, X_train):
    representative_digit_id = []
    X_digits_dist = model.fit_transform(X_train)
    representative_digit_id = np.argmin(X_digits_dist, axis=0)
    X_representatives = X_train.reshape((-1,1))[representative_digit_id]
    return X_representatives, X_representatives.astype(int)

# Apply labels to entire cluster of representative
def propagate_label_data(X_train, y_rep, model, k=50):
    y_train_p = np.empty(len(X_train), dtype=np.int32)
    for i in range(k):
        y_train_p[model.labels_ == i] = y_rep[i]
    return y_train_p

# Partially labels unsupervised data based on some labels found and a given percentage
def partially_label_data(X_train, y_train_p, model, n_k=50, percentile_=20):
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
    y_train_partially_propagated = y_train_p[partially_propagated]
    return X_train_partially_propagated, y_train_partially_propagated
