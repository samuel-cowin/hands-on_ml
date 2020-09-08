from linear_utils import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets

# Define simluated data to perform regression on
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1) 
# Additional column of ones indicating the multiple of the coefficent
X_b = np.c_[np.ones((100,1)), X]

# Perform linear regression and print out the intercept, coefficients, and the prediction of the new data
# This built in function is based on the Sigular Value Decomposition of the data and the resulting pseduoinverse
# This prevents errors when the data is non-invertible and is more computationally efficient
lin_reg = LinearRegression()
lin_reg.fit(X, y)
X_new = np.array([[0], [2]])
print('Linear Regression Prediction: \n{}\n'.format(lin_reg.predict(X_new)))
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print('Theta value SVD: \n{}\n'.format(theta_best_svd))
# The method above is useful for few features or small datasets, but as this grows an alternative method needs to be used
# One such method is gradient descent 
theta = batch_gd_MSE(X_b, y, n_iterations=10000, m=100, eta=0.1, debug=False)
print('Theta value Batch GD: \n{}\n'.format(theta))
# One main issue with the batch method above is that the entire training set is utilized on each pass
# To circumvent this, Stochastic Gradient Descent with a learning schedule can be implemented
theta = stoch_gd_MSE(X_b, y, n_epochs=50, m=100, h1=5, h2=50, debug=False)
print('Theta value Stochastic GD: \n{}\n'.format(theta))
# sklearn has an out of box implementation of this method as well
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
print('SGD Regression Prediction: \n{}\n'.format(sgd_reg.predict(X_new)))
# An optimization that is made over both Batch and Stochastic Gradient Descent is done with mini-batch combining the two approaches
# To do this, you enlarge the size of the sample drawn from the entire training data
theta = minibatch_gd_MSE(X_b, y, n_epochs=50, m=100, b=5, h1=5, h2=50, debug=False)
print('Theta value Mini-Batch GD: \n{}\n'.format(theta))
# An easy extension of the simple linear regression into a non-linear application is using the polynomial regression
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * pow(X, 2) + X + 2 + np.random.randn(m, 1)
# Incorporate different powers as well as interaction terms into the feature space
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
# Run linear regression with the extended feature space
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print('Polynomial Regression Intercept: {} and Coefficients: {}\n'.format(lin_reg.intercept_, lin_reg.coef_))

# Examining the performance of various models on the data
# Linear
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
# Degree 10
poly_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_reg, X, y)

# Another model implementation that is used in the bias-variance trade-off is regularization
# This introduces a penalty term in order to keep the coefficients as small as possible
# Increasing the weight of this penalty will result in flatter curves which reduce overfitting for added bias
# Ridge using MSE minimum computation
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
rp = ridge_reg.predict([[1.5]])
print('Ridge prediction from computation: {}'.format(rp))
# Ridge using SGD with l2 norm penalty (Ridge)
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sp = sgd_reg.predict([[1.5]])
print('Ridge prediction from SGD: {}'.format(sp))
# Lasso using MSE minimum computation
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lp = lasso_reg.predict([[1.5]])
print('Lasso prediction from computation: {}'.format(lp))
# Lasso using SGD with l1 norm penalty (Lasso)
sgd_reg = SGDRegressor(penalty="l1")
sgd_reg.fit(X, y.ravel())
sp = sgd_reg.predict([[1.5]])
print('Lasso prediction from SGD: {}'.format(sp))
# Elastic net provides a good mix between the two and is preferred over Lasso 
# # especially when the number of parameters is high compared with training data or there are correlations between parameters
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
ep = elastic_net.predict([[1.5]])
print('ElasticNet prediction from computation: {}'.format(ep))
# One final regularization method is the early stopping method - stopping when a minimum in cost is detected in training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
poly_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler()),
])
early_stop_MSE(X_train, y_train, X_val, y_val, poly_reg)

# Logistic regression
# Utilizes the scores from linear regression and passes them through the logistic function in order to output scores from 0 to 1
# These scores and corresponding probabilities are used to classify the data point into a binary class system
iris = datasets.load_iris()
X = iris["data"][:, 3:] # petal width for this example
y = (iris["target"] == 2).astype(np.int) # identify specific iris
log_reg = LogisticRegression()
log_reg.fit(X, y)
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = log_reg.predict_proba(X_new)
plt.plot(X_new, y_prob[:, 1], 'g')
plt.plot(X_new, y_prob[:, 0], 'b')
plt.show()
log_pred = log_reg.predict([[1.7], [1.5]])
print('Logistic Regression Prediction: {}'.format(log_pred))
# Multiple classes over a single prediction can be used with the Softmax Regression, one-versus-all, etc.
# The cost function used is the cross-entropy
X = iris["data"][:, (2,3)]
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y) 
s_pred = softmax_reg.predict([[5,2]])
s_prob = softmax_reg.predict_proba([[5,2]])
print("Sofmax Prediction of {} and Probabilities {}\n".format(s_pred, s_prob))