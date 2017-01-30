import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import Adam, SGD
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import pickle

import matplotlib.pyplot as plt

# -------------------------------------
# Data preparation
# -------------------------------------
data_file = "./preprocessed.pickle"
with open(data_file, mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['train_dataset'], data['train_labels']
#X_valid, y_valid = data['valid_dataset'], data['valid_labels']
#X_test, y_test = data['test_dataset'], data['test_labels']

# -------------------------------------
# Cover of NVidia end-to-end network
# -------------------------------------
model = Sequential()
# layer 1, conv
model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape=(66, 200, 3)))
model.add(Activation('relu'))
# layer 2, conv
model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
# layer 3, conv
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
# layer 4, conv
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
# layer 4, conv
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
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
model.add(Dropout(0.5))
# layer 8, fc
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# layer output
model.add(Dense(1))


# -------------------------------------
# Compile and train the model
# -------------------------------------
#model.load_weights('nvidia_net_weights.h5')
opt = SGD(lr=1e-6)
model.compile(opt, 'mse', ['accuracy'])
model.summary()

history = model.fit(X_train, y_train, nb_epoch=300, validation_split=0.0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

# -------------------------------------
# Saving the results
# -------------------------------------
model.save_weights('nvidia_net_weights.h5')

# -------------------------------------
# Evaluate the trained model 
# -------------------------------------
print(model.evaluate(X_train, y_train, verbose=1))
