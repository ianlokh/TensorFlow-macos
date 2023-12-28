import numpy as np
import tensorflow as tf
from tensorflow import keras


def set_gpu(gpu_ids_list):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpus_used = [gpus[i] for i in gpu_ids_list]
            tf.config.set_visible_devices(gpus_used, 'GPU')
            for gpu in gpus_used:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


set_gpu([0])

BATCH_SIZE = 1024
EPOCHS = 100

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape([x_train.shape[0], x_train.shape[1], x_train.shape[2], 1])
x_test = x_test.reshape([x_test.shape[0], x_test.shape[1], x_test.shape[2], 1])

x_train = x_train / 255
x_test = x_test / 255

from tensorflow.keras.utils import to_categorical

number_of_classes = 10
y_train = to_categorical(y_train, number_of_classes)
y_test = to_categorical(y_test, number_of_classes)

# Model Buidling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
