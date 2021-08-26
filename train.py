import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import os
from keras.datasets import cifar10

(x_train, y_train),  (x_test, y_test) = cifar10.load_data()
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train = x_train / 255
x_test = x_test / 255

model = Sequential()
model.add( Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)) )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add( Conv2D(32, (5,5), activation='relu') )
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checpoint_dir = os.path.dirname(checkpoint_path)
cp_callbacks = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model.load_weights(checkpoint_path)

EPOCHS = 40
hist = model.fit(x_train, y_train_one_hot,
                 batch_size = 256,
                 epochs = EPOCHS,
                 validation_split = 0.2,
                 callbacks=[cp_callbacks])

model.evaluate(x_test, y_test_one_hot)[1]
