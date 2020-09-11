from sklearn.datasets import fetch_openml
from dim_reduce_utils import svd_components, n_components_projection, components_from_variance, train_test_split, kcpa_scorer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt

# In order to implement principal components analysis, singular value decomposition is used to determine which vectors the components lie on
n_c = 20
X = np.random.rand(100,n_c)
X[0] = X[0] * 10
X[1] = X[1] * 8
V, X_c = svd_components(X, print_=False)
X_p = n_components_projection(X_c, V, print_=False)

# PCA (SVD) can also be done through sklearn
pca = PCA(n_components=n_c, svd_solver='full')
X_p_sk = pca.fit_transform(X)
print('Explained variance per component: {}'.format(pca.explained_variance_ratio_))

# Selecting number of components based on the percentage of variance explained
ratio = 0.9
num_c = 169
d = components_from_variance(X, ratio, PCA(), n_components=n_c, plot=False)
print('Number of components needed for {}% variance explained is {}'.format(ratio*100, d))

# Compression and recovery using PCA
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, y_train, X_test, y_test = train_test_split(60000, X, y)
num_batches = 100
inc_pca = IncrementalPCA(n_components=num_c)
for X_batch in np.array_split(X_train, num_batches):
    inc_pca.partial_fit(X_batch)
X_recovered = inc_pca.inverse_transform(inc_pca.transform(X_train))
# plt.imshow(X_train[1].reshape(28,28))
# plt.show()
# plt.imshow(X_recovered[1].reshape(28,28))
# plt.show()

# Utilizing KernelPCA to apply kernel transformations for dimensionality reduction
kcpa = KernelPCA(n_components=num_c, kernel="rbf", fit_inverse_transform=True, gamma=0.04)
X_preimage = kcpa.inverse_transform(kcpa.fit_transform(X_train))
plt.imshow(X_preimage[1].reshape(28,28))
plt.show()