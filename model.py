from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# -------------------------------------
# Data preparation
# -------------------------------------


# -------------------------------------
# Cover of NVidia end-to-end network
# -------------------------------------
model = Sequential()
# layer 1, conv
model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape=(200, 66, 3)))
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
model.compile('adam', 'mse', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, nb_epoch=10, validation_split=0.2)

# -------------------------------------
# Saving the results
# -------------------------------------
model.summary()
model.save_weights('nvidia_net_weights.h5')
