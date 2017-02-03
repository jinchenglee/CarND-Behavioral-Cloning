import csv
import numpy as np
import cv2
import pickle
import os
import sys
import tables
import matplotlib.pyplot as plt

# -------------------------------------
# Command line argument processing
# -------------------------------------
if len(sys.argv) < 2:
    print("Missing training data file.")
    print("python3 model.py <data.h5>")

H5_FILE = str(sys.argv[1])


# ------------------
# Read data from preprocessed HDF5 file
# ------------------
f = tables.open_file(H5_FILE, 'r')

# Convert to numpy array
angle = f.root.steer
throttle =f.root.throttle
brake = f.root.brake
speed = f.root.speed

plt.figure(1)

plt.subplot(131)
plt.plot(angle, throttle, 'ro')
plt.xlabel('angle')
plt.ylabel('throttle')

plt.subplot(132)
plt.plot(angle, speed, 'ro')
plt.xlabel('angle')
plt.ylabel('speed')

plt.subplot(133)
plt.hist(np.array(angle),20)
plt.xlabel('angle')
plt.ylabel('number of samples')

plt.tight_layout()
plt.suptitle(H5_FILE, size=15)
plt.subplots_adjust(top=0.9)
plt.show()


