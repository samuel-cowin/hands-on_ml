from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons, fetch_openml, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from clustering_utils import train_test_split_index, partially_label_data, plot_training_model, labels_for_data, propagate_label_data
import matplotlib.pyplot as plt
import numpy as np

# Unsupervised learning techniques are used on data that is not labeled in order to draw conclusions from the data
# K-means clustering is a popular technique in which you take k clusters and iterate between finding the centroid of these points and labeling closest points

# Generate data
X,y = make_moons(n_samples=1000, noise=0.15, random_state=42)
k=int(input('Number of clusters: '))
percent = int(input('Percentage of data to be labeled from clusters: '))

# Implement KMeans Clustering, typically scaling is done first for irregular datasets
mb_k_clus = MiniBatchKMeans(n_clusters=k, n_init=10)
y_pred = mb_k_clus.fit_predict(X)
centers = mb_k_clus.cluster_centers_
# For dimensionality reduction, applying the distances from centroids as a transform can replace the feature space
x_dist = mb_k_clus.transform(X)
# To determine if the clustering method worked well, using silhouette score (closeness to its centroid and away from others) works better than inertia
s_score = silhouette_score(X, mb_k_clus.labels_)
# Applying this to each of the points within a cluster for a silhouette diagram tells you how well the clusers work and if the sizes are similar

# Color Segmemtation
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(np.uint8)
X_train, y_train, X_test, y_test = train_test_split_index(60000, X, y)
plt.imshow(X_train[1].reshape(28,28))
plt.show()
X_train.reshape(-1,3)
kmeans = KMeans(n_clusters=k).fit(X_train)
seg_img = kmeans.cluster_centers_[kmeans.labels_]
plt.imshow(seg_img[1].reshape(28,28))
plt.show()

# Dimensionality reduction for supervised classification
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)
print('Baseline Logistic Regression Score: {}'.format(log_reg.score(X_test, y_test)))
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=k)),
    ("log_reg", LogisticRegression(C=0.5, max_iter=10000)),
])
pipeline.fit(X_train, y_train)
print('Logistic Regression Score after clustering: {}'.format(pipeline.score(X_test, y_test)))
# Method for finding the optimal number of clusters
param_grid = dict(kmeans__n_clusters=range(2,150))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
print('Logistic Regression Score after best clustering: {}'.format(grid_clf.score(X_test, y_test)))

# Semi-supervised learning
kmeans = KMeans(n_clusters=k)
X_rep, y_rep = labels_for_data(kmeans, X_train)
X_rep_fix = np.array(X_rep.reshape(1,-1))
y_rep_fix = np.array(y_rep.reshape(1,-1))
y_train_p = propagate_label_data(X_train, y_rep, kmeans, k=50)
X_pp, y_pp = partially_label_data(X_train, y_train_p, kmeans, k, percent)
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_rep, y_rep)
print('Logistic Regression Score after semi-supervised learning: {}'.format(log_reg.score(X_test, y_test)))

# Another clustering mechanism is the DBscan algorithm
# This can be used to train other classification algorithms for heightened performance
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
plot_training_model(X, y, knn)