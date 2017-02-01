import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import Adam, SGD
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import *

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import sys

# -------------------------------------
# Command line argument processing
# -------------------------------------
if len(sys.argv) < 2:
    print("Missing training data file.")
    print("python3 model.py <data.pickle>")

data_file = str(sys.argv[1])

# -------------------------------------
# Data preparation
# -------------------------------------
with open(data_file, mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['train_dataset'], data['train_labels']
#X_valid, y_valid = data['valid_dataset'], data['valid_labels']
#X_test, y_test = data['test_dataset'], data['test_labels']
#X_train, y_train = shuffle(X_train, y_train)

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
# layer 3, conv
model.add(Convolution2D(64, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
# layer 4, conv
model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))
# layer 4, conv
model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))
# Flatten
model.add(Flatten())
# layer 5, fc
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.6))
# layer 6, fc
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.6))
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
opt = Adam(lr=0.0002)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, nb_epoch=10, batch_size=64, validation_split=0.2)
# list all data in history
print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['acc'])
##plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
#plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
#plt.show()

# -------------------------------------
# Saving the results
# -------------------------------------
model.save_weights('model.h5')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# -------------------------------------
# Evaluate the trained model 
# -------------------------------------
for i in range(50):
    print(y_train[i], float(model.predict(X_train[i].reshape([1,66,200,3]), batch_size=1)))
