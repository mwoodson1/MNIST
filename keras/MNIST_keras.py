import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Set random seed
np.random.seed(1234)

#Set parameters
nb_classes = 10
batch_size = 256
nb_epoch = 10

#Preprocess data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Create network
model = Sequential()

#32 conv filters using 5x5 kernels and relu activation
model.add(Convolution2D(32,1,5,5,border_mode='same'))
model.add(Activation('relu'))

#32 conv filters using 3x3 kernels and relu activation
model.add(Convolution2D(32,32,3,3,border_mode='same'))
model.add(Activation('relu'))

#Max pooling and 50% Dropout
model.add(MaxPooling2D(poolsize=(2,2)))
model.add(Dropout(.5))

#16 conv filters using 3x3 kernels and relu activation
model.add(Convolution2D(16,32,3,3,border_mode='same'))
model.add(Activation('relu'))

#50% Dropout
model.add(Dropout(.5))

#Flatten and add dense network
model.add(Flatten())
model.add(Dense(16*196, 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')#, theano_mode='DebugMode')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#Display example images and their labels