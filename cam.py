
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import Adam, SGD
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import *
from keras import __version__ as keras_version

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tables
import sys
import cv2
import scipy
import argparse
import h5py
import shutil

K.set_learning_phase(0) # All operations in test mode

EPSILON = 1e-7
CONV_LAYER = 'convolution2d_4'

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
    '''
    Return the gradient value, but when angle is very small, we care about those values
    contribute to the result of its being small, thus the last "else" of 1/x. 

    TODO: It seems this method cannot deal with negative values well. Say in small angle
    case that postive and negative values average out each other.
    '''
    threshold_degree = 3.0
    if angle > threshold_degree * scipy.pi / 180.0:
        return x
    elif angle < -threshold_degree * scipy.pi / 180.0:
        return -x
    else:
        # Avoid div-by-0.
        x = x + EPSILON
        return (1.0/x) * np.sign(angle)


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def visualize_class_activation_map(gradients_function, img_path, output_path):
    print('DEBUG_5: ', img_path)
    original_img = cv2.imread(img_path, 1)
    width, height, _ = original_img.shape

    # !!!IMPORTANT!!!
    # Recorded image using OpenCV (BGR), convert to RGB before feeding into network.
    # The replayed image has predicted angle a little bit different from recorded value, this is
    # probably because img compression/de-compression in recording and replay.
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    img = normalize_color(img_rgb[None,:,:,:])
    img = img.reshape([1,66,200,3])

    interested_layer_outputs, grads_val, angle = gradients_function([img])
    print('DEBUG_6: ', interested_layer_outputs.shape)
    print('DEBUG_7: ', grads_val.shape)

    # Sanity check angle vs. directly prediction from model steering_angle. They should match.
    #steering_angle = float(model.predict(img, batch_size=1))
    #print("predicted angle = ", angle)
    #print("steering_angle = ", steering_angle)

    #class_weights = np.mean(grads_val, axis=(0,1))
    # Evaluate the angle to determine the weights
    class_weights = grad_cam_loss(grads_val, angle)
    print('DEBUG_8 class_weights.shape =',class_weights.shape)

    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = interested_layer_outputs.shape)
    # Element-wise muliplication
    cam = class_weights*interested_layer_outputs
    print("DEBUG_9 cam.shape = ", cam.shape)
    # Average among the number of filters in this layer
    cam = np.mean(cam, axis = (2))
    print("DEBUG_10 cam.shape = ", cam.shape)

    #Bug? Should use abs(cam) before scaling to colormap
    #cam /= np.max(cam)
    cam /= np.max(np.abs(cam))

    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap*0.5 + original_img
    cv2.putText(img,str(angle),(50,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.imwrite(output_path, img)

# -------------------------------------
# Restore cover of NVidia end-to-end network
# -------------------------------------

import os
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradients Activation Mapping processing.')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
            ', but the model was built using ', model_version)
        
    model = load_model(args.model)
    model.summary()
    
    ###------ Key section of assistance functions definition ------

    # Get the final output of the model
    pred_angle = K.sum(model.layers[-1].output)
    print('DEBUG_1: ', model.layers[-1].output)
    print('DEBUG_2: pred_angle', pred_angle)

    # The specific layer of interest to back-annotate the activation
    interested_layer = get_output_layer(model, CONV_LAYER)

    # Gradients from output to interested layer, not sure why it is output as a list [blah], 
    #   thus the [0] at the end
    grads = normalize(K.gradients(pred_angle, interested_layer.output)[0])
    print('DEBUG_3: ', K.gradients(pred_angle, interested_layer.output))
    print('DEBUG_4: ', K.gradients(pred_angle, interested_layer.output)[0])

    # Feed input, grab output (pred_angle), interested layer output and gradients from 
    #   pred_angle back to interested layer. 
    gradients_function = K.function([model.layers[0].input], [interested_layer.output[0], grads[0], pred_angle])
    ###------------------------------------------------


    if args.image_folder != '':
        print("Work with image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            print("Image folder doesn't exist!")
        else:
            in_folder = args.image_folder
            out_folder = in_folder + '_cam'
            print("Creating image folder at {}".format(out_folder))
            if os.path.exists(out_folder):
                shutil.rmtree(out_folder)
            os.makedirs(out_folder)

            for img_file in glob.glob(in_folder+"/*.png"):
                print(img_file)
                in_folder_str_len = len(in_folder)
                trim_img_file = img_file[in_folder_str_len:-4]
                visualize_class_activation_map(gradients_function, img_file, out_folder+'/'+trim_img_file+'_'+CONV_LAYER+'.cam.jpg')
    else:
        print("Where are the images stored? Please provide image folder.")


