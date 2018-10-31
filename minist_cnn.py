'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

'''
from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns


import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator



batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# x_train = x_train[0:10000,:,:,:]
# y_train = x_train[0:10000,:,:,:]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#2 --- > 1
#3 ----> 0
#61 -- > 8

# for i in range(100):
#     print(i)
#     plt.imshow(x_test[i][:,:,0]);

#     plt.show()

# plt.imshow(x_test[11][:,:,0]);

# plt.show()


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if os.path.exists('keras_model.h5'):
    print('Loading model...')
    model = load_model('keras_model.h5')
else:
    print('Building model...')
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(25, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16,(3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
model.save('keras_model.h5')

print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(model.summary())

from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)


weight_conv2d_1 = model.layers[0].get_weights()[0][:,:,0,:]

col_size = 4
row_size = 4
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(12,8))
plt.suptitle('Convolutional Layer 1')
for row in range(0,row_size):
  for col in range(0,col_size):
    ax[row][col].imshow(weight_conv2d_1[:,:,filter_index],cmap="gray")
    filter_index += 1
plt.show()

weight_conv2d_2 = model.layers[1].get_weights()[0][:,:,0,:]

col_size = 5
row_size = 5
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(12,8))
plt.suptitle('Convolutional Layer 2')
for row in range(0,row_size):
  for col in range(0,col_size):
    ax[row][col].imshow(weight_conv2d_2[:,:,filter_index],cmap="gray")
    filter_index += 1
plt.show()

weight_conv2d_3 = model.layers[3].get_weights()[0][:,:,0,:]
col_size = 4
row_size = 4
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(12,8))
plt.suptitle('Convolutional Layer 3')
for row in range(0,row_size):
  for col in range(0,col_size):
    ax[row][col].imshow(weight_conv2d_3[:,:,filter_index],cmap="gray")
    filter_index += 1
plt.show()


def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
activations_0 = activation_model.predict(x_test[3].reshape(1,28,28,1))
display_activation(activations_0, 4, 4, 3)
output = activations_0[9]
print(output)
plt.suptitle('Feture map for digit 1')
plt.show()

activations_8 = activation_model.predict(x_test[61].reshape(1,28,28,1))
display_activation(activations_8, 4, 4, 3)
output = activations_8[9]
print(output)
plt.suptitle('Feture map for digit 8')
plt.show()


activations_1 = activation_model.predict(x_test[2].reshape(1,28,28,1))
output = activations_1[9]
print("Predict 1", output)


x_left = x_test[2].reshape(1,28,28,1)
x_left = x_left[:,:,0:26,:]
plt.imshow(x_left[0,:,:,0], cmap = 'gray');
plt.show()
print(x_left.shape)
x_left = np.pad(x_left, ((0, 0), (0, 0), (0, 2), (0, 0)), 'edge')
print(x_left.shape)
plt.imshow(x_left[0,:,:,0]);

activations_1_left = activation_model.predict(x_left)
output = activations_1_left[9]
print("left shift", output)

x_right = x_test[2].reshape(1,28,28,1)
x_right = x_left[:,:,2:28,:]
plt.imshow(x_right[0,:,:,0], cmap = 'gray');
plt.show()
print(x_right.shape)
x_right = np.pad(x_right, ((0, 0), (0, 0), (2, 0), (0, 0)), 'edge')
print(x_right.shape)
plt.imshow(x_right[0,:,:,0], cmap = 'gray');
activations_1_right = activation_model.predict(x_right)
output = activations_1_right[9]
print("right shift", output)

plt.show()


#print(x_test[11])
activations_1 = activation_model.predict(x_test[11].reshape(1,28,28,1))

map_1 = np.zeros(23 * 23)
map_2 = np.zeros(23 * 23)
map_3 = np.zeros(23 * 23)


for i in range(23):
    for j in range(23):
        x = x_train[18].reshape(1,28,28,1).copy()
        for k in range(6):
            for l in range(6):
                x[0, i + k, j + l, 0] = 0
        if i == 15 and j == 10:
            plt.imshow(x[0,:,:,0],cmap = 'gray')
            plt.show()
        activations_6 = activation_model.predict(x)
        output = activations_6[9]
        map_1[i * 23 + j] = output[0,6]
        map_2[i * 23 + j] = output[0, np.argmax(output)]
        map_3[i * 23 + j] = np.argmax(output)

map_1 = map_1.reshape(23,23)
map_2 = map_2.reshape(23,23)
map_3 = map_3.reshape(23,23)

print(map_1)
print(map_2)
print(map_3)

plt.imshow(map_1, cmap = 'gray')
plt.title('Map 1')
plt.show()
plt.imshow(map_2, cmap = 'gray')
plt.title('Map 2')
plt.show()
plt.imshow(map_3, cmap = 'gray')
plt.title('Map 3')
plt.show()
