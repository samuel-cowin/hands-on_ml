import tensorflow as tf
from tensorflow import keras

# Model creation through subclassing
class basic_nn(keras.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(units, activation=activation)
        self.output = keras.layers.Dense(1)

    def call(self, input_):
        input_ = input_
        hidden = self.hidden(input_)
        output = self.output(hidden)
        return output

# Build generic model from passed hyperparameters
def build_nn(n_hidden=1, n_neurons=30, learning_rate=0.001, input_shape=[8], activation='relu', loss='mse'):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation=activation))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)
    return model