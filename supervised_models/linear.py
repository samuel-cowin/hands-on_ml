from sklearn.linear_model import LinearRegression
from linear_utils import stoch_gd_MSE, batch_gd_MSE, minibatch_gd_MSE
from sklearn.linear_model import SGDRegressor
import numpy as np

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