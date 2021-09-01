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

def translate(i):
    switcher={
        'airplane':' ett flygplan.',
        'automobile':' en bil.',
        'bird':' en fågel.',
        'cat':' en katt.',
        'deer':' ett rådjur.',
        'dog':' en hund.',
        'frog':' en groda.',
        'horse':' en häst.',
        'ship':' ett skep.',
        'truck':' en lastbil.'
    }
    return switcher.get(i,"Vet ej")

def predict(new_image):
    from skimage.transform import resize
    resized_image = resize(new_image, (32,32,3))
    img = plt.imshow(resized_image)
    predictions = model.predict(np.array([resized_image]))

    list_index = [0,1,2,3,4,5,6,7,8,9]
    x = predictions
    for i in range(10):
        for j in range(10):
            if x[0][list_index[i]] > x[0][list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    
    
    print(classification[list_index[0]], ':', round(predictions[0][list_index[0]]))
    result = ""

    if predictions[0][list_index[0]] > 0.95:
        result += "Bilden är av"
    elif predictions[0][list_index[0]] > 0.75:
        result += "Tror det är"
    else:
        result += "Vet inte kanske"

    result += translate(classification[list_index[0]])


    return result