from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np


# Plotting learning curves for visualization of performance
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label = 'train')
    plt.plot(np.sqrt(val_errors), 'b', linewidth=2, label = 'val')
    plt.axis([0, 80, 0, 3])
    plt.show()

# Random sample of training data incorporating batches
def random_sample_train(X, y, m, b, debug=False):
    r_index = np.random.randint(m)
    xi = X[r_index:r_index+1]
    yi = y[r_index:r_index+1]
    for i in range(1,b):
        if debug:
            print(i)
        r_index = np.random.randint(m)
        xi = np.vstack((xi, X[r_index:r_index+1]))
        yi = np.vstack((yi, y[r_index:r_index+1]))
    return xi, yi

# Learning rate calculation from training data seen up to current point
def learning_schedule(t_data, h1, h2):
    return h1 / (t_data + h2)

# Gradient Descent learning update
def GD_update(X, y, theta, eta, m):
    gradients_u = 2/m * np.transpose(X).dot(X.dot(theta)-y)
    theta_u = theta-eta*gradients_u
    return gradients_u, theta_u

# Batch Gradient Descent for MSE cost function
def batch_gd_MSE(X, y, n_iterations=1000, m=100, eta=0.1, debug=False):
    theta = np.random.randn(2,1)
    for iteration in range(n_iterations):
        _, theta = GD_update(X, y, theta, eta, m)
        if debug:
            print('Number of iterations completed is {}'.format(iteration))
    return theta

# Stochastic Gradient Descent for MSE cost function
def stoch_gd_MSE(X, y, n_epochs=50, m=100, h1=5, h2=50, debug=False):
    theta = np.random.randn(2,1)
    for e in range(n_epochs):
        for i in range(m):
            # Find random training value
            x_r, y_r = random_sample_train(X, y, m, 1)
            eta = learning_schedule(e * m + i, h1, h2)
            _, theta = GD_update(x_r, y_r, theta, eta, 1)
    return theta

# Mini-batch Gradient Descent for MSE cost function
def minibatch_gd_MSE(X, y, n_epochs=50, m=100, b=5, h1=5, h2=50, debug=False):
    theta = np.random.randn(2,1)
    for e in range(n_epochs):
        for i in range(int(m/b)):
            # Find random training value
            x_r, y_r = random_sample_train(X, y, m, b)
            for j in range(b):
                eta = learning_schedule(e * m + i * b + j, h1, h2)
                _, theta = GD_update(np.asarray([x_r[j]]), np.asarray([y_r[j]]), theta, eta, int(m/b))
    return theta

# Early stopping for gradient descent on linear models
def early_stop_MSE(X_train, y_train, X_val, y_val, model, gd_method=SGDRegressor, n_epochs=1000, min_val_error=float("inf")):
    X_train_scaled = model.fit_transform(X_train)
    X_val_scaled = model.transform(X_val)

    sgd_reg = gd_method(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)

    best_epoch = None
    best_model = None
    for e in range(n_epochs):
        sgd_reg.fit(X_train_scaled, y_train.ravel())
        y_val_predict = sgd_reg.predict(X_val_scaled)
        val_error = mean_squared_error(y_val, y_val_predict)
        if val_error < min_val_error:
            min_val_error = val_error
            best_epoch = e
            best_model = clone(sgd_reg)
    return best_epoch, best_model