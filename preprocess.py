import csv
import numpy as np
import cv2
import pickle
import os
import sys

# -------------------------------------
# Command line argument processing
# -------------------------------------
if len(sys.argv) < 2:
    print("Missing training data file.")
    print("python3 model.py <data.pickle>")

DATA_DIR = str(sys.argv[1])


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
    Normalize the image data on per channel basis. 
    """
    img_normed_color = np.zeros_like(image_data, dtype=float)
    for ch in range(image_data.shape[3]):
        tmp = normalize_grayscale(image_data[:,:,:,ch])
        img_normed_color[:,:,:,ch] = tmp
    #print(np.max(img_normed_color))
    #print(np.min(img_normed_color))
    return img_normed_color

# ------------------
# Read data from recorded dataset
# ------------------
X_train = []
y_train = []

SMALL_SCALE = 0.9
LARGE_SCALE = 1.1

i = 0
with open(DATA_DIR+'/driving_log.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'center':
            # center
            img = cv2.imread(DATA_DIR+"/"+row[0])
            img_crop = img[56:160,:,:]
            img_resize = cv2.resize(img_crop, (200,66))
            angle = np.float32(row[3])
            X_train.append(img_resize)
            y_train.append(angle)

            if angle > 0: # right turn
                l_angle = LARGE_SCALE * angle
                r_angle = SMALL_SCALE * angle
            else: # left turn
                l_angle = SMALL_SCALE * angle
                r_angle = LARGE_SCALE* angle

            # left
            img = cv2.imread(DATA_DIR+"/"+row[1])
            img_crop = img[56:160,:,:]
            img_resize = cv2.resize(img_crop, (200,66))
            X_train.append(img_resize)
            y_train.append(l_angle)

            # right
            img = cv2.imread(DATA_DIR+"/"+row[2])
            img_crop = img[56:160,:,:]
            img_resize = cv2.resize(img_crop, (200,66))
            X_train.append(img_resize)
            y_train.append(r_angle)

            i = i +1
            print(i)

#<<JC>> i = 0
#<<JC>> with open(DATA_DIR+'/driving_log.csv', newline='') as f:
#<<JC>>     reader = csv.reader(f)
#<<JC>>     for row in reader:
#<<JC>>         if row[0] != 'center':
#<<JC>>             img = cv2.imread(DATA_DIR+"/"+row[0])
#<<JC>>             img_crop = img[56:160,:,:]
#<<JC>>             img_resize = cv2.resize(img_crop, (200,66))
#<<JC>>             angle = np.float32(row[3])
#<<JC>>             # Horizontally flipped version of the image
#<<JC>>             img_resize_flip = cv2.flip(img_resize,0)
#<<JC>>             X_train.append(img_resize_flip)
#<<JC>>             y_train.append(-angle)
#<<JC>> 
#<<JC>>             if angle > 0: # right turn
#<<JC>>                 l_angle = LARGE_SCALE * angle
#<<JC>>                 r_angle = SMALL_SCALE * angle
#<<JC>>             else: # left turn
#<<JC>>                 l_angle = SMALL_SCALE * angle
#<<JC>>                 r_angle = LARGE_SCALE* angle
#<<JC>> 
#<<JC>>             # left
#<<JC>>             img = cv2.imread(DATA_DIR+"/"+row[1])
#<<JC>>             img_crop = img[56:160,:,:]
#<<JC>>             img_resize = cv2.resize(img_crop, (200,66))
#<<JC>>             # Horizontally flipped version of the image
#<<JC>>             img_resize_flip = cv2.flip(img_resize,0)
#<<JC>>             X_train.append(img_resize_flip)
#<<JC>>             y_train.append(-l_angle)
#<<JC>> 
#<<JC>>             # right
#<<JC>>             img = cv2.imread(DATA_DIR+"/"+row[2])
#<<JC>>             img_crop = img[56:160,:,:]
#<<JC>>             img_resize = cv2.resize(img_crop, (200,66))
#<<JC>>             # Horizontally flipped version of the image
#<<JC>>             img_resize_flip = cv2.flip(img_resize,0)
#<<JC>>             X_train.append(img_resize_flip)
#<<JC>>             y_train.append(-r_angle)
#<<JC>> 
#<<JC>>             i = i +1
#<<JC>>             print(i)


# sanity check
#for i in range(len(X_train)):
#    cv2.imwrite("resized_"+str(i)+".png", X_train[i])

# Convert to numpy array
X_train = normalize_color(np.array(X_train))
y_train = np.array(y_train)
print("max X_train = ", np.max(X_train))
print("min X_train = ", np.min(X_train))
print("y_train = ", y_train)

# -------------------
# Save to pickle file
# -------------------
pickle_file = DATA_DIR+".pickle"
print(pickle_file)
if not os.path.isfile(pickle_file):
    print('Saving preprocessed data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': X_train,
                    'train_labels': y_train,
#                    'valid_dataset': X_valid,
#                    'valid_labels': y_valid,
#                    'test_dataset': X_test,
#                    'test_labels': y_test,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')



