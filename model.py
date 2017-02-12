import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import *
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tables
import sys

# -------------------------------------
# Command line argument processing
# -------------------------------------
if len(sys.argv) < 2:
    print("Missing training data file.")
    print("python3 model.py <data.h5> <epoch_cnt>")

H5_FILE = str(sys.argv[1])

EPOCH = 5
if len(sys.argv) >2:
    EPOCH = int(sys.argv[2])

# ------------------
# Read data from preprocessed HDF5 file
# ------------------
f = tables.open_file(H5_FILE, 'r')

# -------------------------------------
# Data preparation
# -------------------------------------

X_train = np.array(f.root.img)
y_train = np.array(f.root.steer)
print(X_train.shape, y_train.shape)
print("Train data[23] mean = ", np.mean(X_train[23]))

X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.2, random_state=88
                )
#X_train, y_train = shuffle(X_train, y_train)
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

train_datagen = ImageDataGenerator(
            )
#train_datagen = ImageDataGenerator(
#            rotation_range=5,
#            height_shift_range=0.1,
#            shear_range= 0.1,
#            zoom_range = 0.1,
#            fill_mode = 'nearest'
#          )
train_datagen.fit(X_train)

val_datagen = ImageDataGenerator(
            )
val_datagen.fit(X_valid)


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
model.load_weights('weights.h5')
#opt = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=0.00007)
#opt = RMSprop(lr=0.00008)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit_generator(
                # ==== Unmask below line to dump image out to take snapshot of what's being fed into training process.
                #train_datagen.flow(X_train, y_train, batch_size=64,save_to_dir="./", save_prefix="fitgen_", save_format="png"), 
                # ==== Use below line to do normal training
                train_datagen.flow(X_train, y_train, batch_size=64), 
                samples_per_epoch=X_train.shape[0], 
                nb_epoch=EPOCH,
                validation_data=val_datagen.flow(X_valid, y_valid, batch_size=64), 
                nb_val_samples=X_valid.shape[0]
                )

# -------------------------------------
# Saving the results
# -------------------------------------
model.save_weights('weights.h5')
model.save('model.h5')


