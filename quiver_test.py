import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import Adam, SGD
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tables
import sys

from quiver_engine import server


# -------------------------------------
# Cover of NVidia end-to-end network
# -------------------------------------
model = Sequential()
# layer 1, conv
model.add(Convolution2D(36, 5, 5, subsample=(2,2), input_shape=(66, 200, 3)))
model.add(Activation('relu'))
# layer 2, conv
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 3, conv
model.add(Convolution2D(64, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 4, conv
model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 4, conv
model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Flatten
model.add(Flatten())
# layer 5, fc
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 6, fc
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer 7, fc
model.add(Dense(50))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
# layer 8, fc
model.add(Dense(10))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
# layer output
model.add(Dense(1))


# -------------------------------------
# Compile and train the model
# -------------------------------------
model.load_weights('model.h5')
#opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.summary()

# -------------------------------------
# Evaluate the trained model 
# -------------------------------------
server.launch(model, input_folder='./bootstrap/IMG')
