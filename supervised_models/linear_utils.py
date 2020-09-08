import numpy as np

# Random sample of training data incorporating batches
def random_sample_train(X, y, m, b):
    r_index = np.random.randint(m)
    xi = X[r_index:r_index+1]
    yi = y[r_index:r_index+1]
    for i in range(1,b):
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