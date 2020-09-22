import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np
from cnn_cv_utils import ResidualUnit, preprocess

# Load images and scale
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape
# Implementation of basic architecture to detect feature maps in image sets
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1
filters[3, :, :, 1] = 1
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
plt.imshow(china, cmap="gray")
plt.show()
plt.imshow(outputs[0, :, :, 1], cmap="gray")
plt.show()

# Retrieving MNIST dataset
# Implementation of straight-forward Keras model for classification of image dataset
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu",
                        padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()

#Implementation of ResNet with Keras and custom Residual Unit
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 7, strides=2, data_format="channels_last", input_shape=(224, 224, 3), padding="same", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filter = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters==prev_filter else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filter=filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

# Using pre-trained model
model = keras.applications.resnet50.ResNet50(weights="imagenet")
images_resize = tf.image.resize(images, [224,224])
inputs = keras.applications.resnet50.preprocess_input(images_resize*255)
Y_proba = model.predict(inputs)
top_k = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print("Image #{}".format(image_index))
    for class_id, name, y_p in top_k[image_index]:
        print(" {}  -   {:12s}  {:.2f}%".format(class_id, name, y_p * 100))
    print()

# Transfer learning with TFDS for Classification and Localization
# The dataset does not support this implementation, but the code is provided as illustration
# Get the data
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
n_classes = info.features["label"].num_classes
# Split the data
test_data = tfds.load("tf_flowers", split="train[:10%]", shuffle_files=True, as_supervised=True)
valid_data = tfds.load("tf_flowers", split="train[10:25]", as_supervised=True)
train_data = tfds.load("tf_flowers", split="train[25:]", as_supervised=True)
# Preprocess the data
batch_size = 32
train_data = train_data.map(preprocess).batch(batch_size).prefetch(1)
valid_data = valid_data.map(preprocess).batch(batch_size).prefetch(1)
test_data = test_data.map(preprocess).batch(batch_size).prefetch(1)
# Generate base_model from Xception and add new output layers
base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = keras.layers.Dense(n_classes, activation="softmax")(avg)
loc_output = keras.layers.Dense(4)(avg)
model = keras.Model(inputs=base_model.input, outputs=[class_output, loc_output])
# Theoretically you want to freeze already trained layers - this will not work with this implementation
for layer in base_model.layers:
    layer.trainable = False
optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
model.compile(loss=["sparse_categorical_crossentropy","mse"], loss_weights=[0.3,0.7], optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_data, epochs=5, validation_data=valid_data)
# Unfreeze once the final layers have had time to train
for layer in base_model.layers:
    layer.trainable = True
optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_data, epochs=50, validation_data=valid_data)