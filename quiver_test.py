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
# Compile and train the model
# -------------------------------------
model=load_model('model.h5')
#opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.summary()

# -------------------------------------
# Evaluate the trained model 
# -------------------------------------
server.launch(model, input_folder='./quiver_img/IMG')
