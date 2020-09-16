import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from simple_nn_utils import build_nn
from scipy.stats import reciprocal
import numpy as np

# Classification example
# Generate the data for training, validation, and testing for classification
fashion_data = keras.datasets.fashion_mnist
(X_train_all, y_train_all), (X_test, y_test) = fashion_data.load_data()
X_valid, X_train = X_train_all[:5000]/255.0, X_train_all[5000:]/255
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
# Example of a functional model for neural networks
input_a = keras.layers.Input(shape=[28, 28], name='wide_in')
flatten = keras.layers.Flatten()(input_a)
hidden1 = keras.layers.Dense(300, activation='relu')(flatten)
hidden2 = keras.layers.Dense(100, activation='relu')(hidden1)
concat = keras.layers.concatenate([flatten, hidden2])
output = keras.layers.Dense(10, activation='softmax')(concat)
model = keras.Model(inputs=[input_a], outputs=[output])
# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])
# Train the model and evaluate the performance on the validation data
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

# Regression example with best parameter search
# Generate the data and scale it
X, y = fetch_california_housing(return_X_y=True)
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
# Fit keras model to sklearn wrapper
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_nn)
param_dist = {
    'n_hidden': [0, 1, 2, 3],
    'n_neurons': np.arange(1, 100).tolist(),
    'learning_rate': [1e-4, 1e-3, 1e-2]
}
rnd_search = RandomizedSearchCV(keras_reg, param_dist, n_iter=10, cv=2)
rnd_search.fit(X_train, y_train, epochs=30, validation_data=(
    X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(patience=10)])
print(rnd_search.best_params_)