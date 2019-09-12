## Plot ad hoc mnist instances
#from keras.datasets import mnist
#import matplotlib.pyplot as plt
## load (downloaded if needed) the MNIST dataset
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
## plot 4 images as gray scale
#plt.subplot(221)
#plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
#plt.subplot(222)
#plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
#plt.subplot(223)
#plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
#plt.subplot(224)
#plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
## show the plot
#plt.show()


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(p = 0.1))
    model.add(Dense(800, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(p = 0.2))
    model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
from tensorflow.python.keras import backend
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=100, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

import numpy as np
from keras.preprocessing import image
import PIL.ImageOps
image_size = 784
test = []
for x in range(1):
    test_image = image.load_img(str(x)+'.png', target_size = (28,28))
    test_image = PIL.ImageOps.invert(test_image)
    test_image = image.img_to_array(test_image)
    test_image = np.moveaxis(test_image,2,0)
    test_image = test_image[0]
    test_image = test_image.reshape(1,image_size).astype('float32')
    test_image/= 255
    test_pred = model.predict(test_image)
    test_pred = test_pred > 0.5
    print(test_pred)
    test.append(test_pred.tolist())
for x in range(len(test)):
    for y in range(10):
        if test[x][0][y] == True:
            print(y, x)
            break