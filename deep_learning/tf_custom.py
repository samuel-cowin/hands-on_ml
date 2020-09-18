import tensorflow as tf
from tensorflow import keras
from tf_custom_utils import softplus, glorot_initializer, l1_regularizor, pos_weights

layer = keras.layers.Dense(30, activation=softplus, kernel_initializer=glorot_initializer,
                           kernel_regularizer=l1_regularizor, kernel_constraint=pos_weights)