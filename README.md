# CarND-Behavioral-Cloning
* Three-image-input
  - Started with three images as input, one with positive steering angle, one with negative and the other one with 0. Started a complete cover of nvidia end-to-end net. 
  - Dropout was added in, the loss didn't drop at all. After removing the dropouts, it started showing life. It makes sense, as dropout is to avoid overfitting. How can I draw "redundant" information from so limited (3 images!) data? :P
  - Started with SGD and grid search on LR. Found SGD is a bit not stable. Changing to Adam, it is much more stable and faster. 
  
* Data provided by Udacity
  - Moved on to training data provided by Udacity. It contains ~8k images.
  - It took me a while to be able to connect to simulator and try my net. Drive.py needs customization. 
  - Training with ~20 epoches, the car can pass first 2-3 curves, then headed off track and into water. I'm still very excited already, :)

* Track1 at 30 mph
  - Steps: 
  - 1. Training from scratch with left/center/right images augmented from udacity dataset. With this alone, car will crash into wall on black bridge on track1.  
  - 2. Train based on step 1 result but with original udacity dataset, not the augmented one. That's it.

* Track1 at 14, 24, 30 mph solid
  - Changed database from pickle to tables(HDF5)
  - Changed to use keras image generator, used random shear, zoom, rotation.
  - Alternating between local recorded track1 and udacity dataset, used 1-2 times of left/right augmented track1 dataset, achieved the stable state.
  - Still failed badly on track2. Has to deal with overfitting. 
