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
import cv2
import scipy

K.set_learning_phase(0) # All operations in test mode

def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    img_max = np.max(image_data)
    img_min = np.min(image_data)
    a = -0.5
    b = 0.5

    img_normed = a + (b-a)*(image_data - img_min)/(img_max - img_min)
    #print(np.max(img_normed))
    #print(np.min(img_normed))
    return img_normed

def normalize_color(image_data):
    """
    Normalize the image data on per channel basis.  """
    img_normed_color = np.zeros_like(image_data, dtype=float)
    for ch in range(image_data.shape[3]):
        tmp = normalize_grayscale(image_data[:,:,:,ch])
        img_normed_color[:,:,:,ch] = tmp
    #print(np.max(img_normed_color))
    #print(np.min(img_normed_color))
    return img_normed_color


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_cam_loss(x, angle):
    if angle > 5.0 * scipy.pi / 180.0:
        return x
    elif angle < -5.0 * scipy.pi / 180.0:
        return -x
    else:
        return (1.0/x) * np.sign(angle)


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def visualize_class_activation_map(model, img_path, output_path):
        original_img = cv2.imread(img_path, 1)
        width, height, _ = original_img.shape

        #Reshape to the network input shape (3, w, h).
        #img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])
        img = normalize_color(original_img[None,:,:,:])
        img = img.reshape([1,66,200,3])
        
        pred_angle = K.sum(model.layers[-1].output)
        final_conv_layer = get_output_layer(model, "convolution2d_5")

        grads = normalize(K.gradients(pred_angle, final_conv_layer.output)[0])
        gradients_function = K.function([model.layers[0].input], [final_conv_layer.output, grads, pred_angle])

        conv_outputs, grads_val, angle = gradients_function([img])
        conv_outputs, grads_val = conv_outputs[0,:], grads_val[0,:,:,:]

        print("predicted angle = ", angle)

        class_weights = np.mean(grads_val, axis=(0,1))
        # Evaluate the angle to determine the weights
        class_weights = grad_cam_loss(class_weights, angle)
        print("class_weights.shape=",class_weights.shape)

        conv_outputs = conv_outputs[0, :, :]
        print("conv_outputs.shape=",conv_outputs.shape)

        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = (1,conv_outputs.shape[0]))
        for i, w in enumerate(class_weights):
                #print("i=",i, "w=",w)
                cam += w * conv_outputs[:,i]

        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img
        #angle_string = "{:.8f}".format(angle)
        cv2.putText(img,str(angle),(50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
        cv2.imwrite(output_path, img)


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
model.load_model('model.h5')
model.summary()

import os
import glob
for img_file in glob.glob("fitgen*.png"):
    print(img_file)
    visualize_class_activation_map(model, img_file, img_file+'.cam.png')

