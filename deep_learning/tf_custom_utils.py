import tensorflow as tf
from tensorflow import keras

# Example of a basic implementation of a custom loss function
def huber_loss(y_true, y_pred):
    error = y_true-y_pred
    is_small = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small, squared_loss, linear_loss)

# Example of loss class that has custom thresholds and can be saved
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true-y_pred
        under_thres = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * \
            tf.abs(error) - pow(self.threshold, 2) / 2
        return tf.where(under_thres, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'threshold': self.threshold}

# Example of custom activation function similar to tf.nn.softplus()
def softplus(z):
    return tf.math.log(tf.exp(z)+1.0)

# Example of a custom initialization function similar to the Keras glorot_normal()
def glorot_initializer(shape, dtype=tf.float32):
    sd = tf.sqrt(2./(shape[0]+shape[1]))
    return tf.random.normal(shape, stddev=sd, dtype=dtype)

# Example of l1 regularization with hyperparameter 0.01
def l1_regularizor(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

# Example of a class implementing l1 regularization in order to save and pass thresholds
class L1Regularizor(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}

# Example of a constraint forcing all weights to be positive much like relu
def pos_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)

# Creating custom metrics
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber = HuberLoss(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber.call(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


# Creating custom layers for layers with no weights,
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
# with weights,
class Dense_layer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    def build(self, batch_input_shape):
        self.kernel = self.add_weight(name="kernel", shape=[
                                      batch_input_shape[-1], self.units], initializer="glorot_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape)
    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "activation": keras.activations.serialize(self.activation)}
# for multiple layers,
class TwoToThreeLayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        return [X1+X2, X1*X2, X1/X2]
    
    def compute_output_shape(self, batch_input_shape):
        b1, _ = batch_input_shape
        return [b1,b1,b1]
# and for modification of behavior in training
class GaussianNoiseLayer(keras.layers.Layer):
    def __init__(self, sd, **kwargs):
        super().__init__(**kwargs)
        self.sd = sd
    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.sd)
            return X + noise
        else:
            return X
    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape