import csv
import numpy as np
import cv2
import pickle
import os

DATA_DIR = './bootstrap'

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

with open(DATA_DIR+'/driving_log.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        img = cv2.imread(row[0])
        img_crop = img[56:160,:,:]
        img_resize = cv2.resize(img_crop, (200,66))
        X_train.append(img_resize)
        y_train.append(np.float32(row[3]))

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
pickle_file = 'preprocessed.pickle'
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



