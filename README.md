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

* Track2 shows sign of life, crashed at a right turn where has light background (Choose fastest mode thus there's no shade in track2)
  - Recorded track2 training data
  - Trained based on previous best model/weight, using smaller learning rate, for two five epochs. Loss at ~0.45.
  - Checked still solid on track 1, speed 11, 20, 30 mph.
  - Track 2 failed at a spot rith right turn that background is bright sky.

* Track2 completed at 30mph. 
  - Recorded the failure scene (right turn with bright background) for several times of passing with right steering angle.
  - Trained with lower lr for 5 epoches, RMSprop optimizer.

* Track2 completed at 20mph w/ shadow turned on.
  - steps:  
    > git checkout model.json model.h5
    > p3 model.py track2_recover.h5.warp_shadow 3

* Track2 completed at 30mph w/ shadow on.
  - Model becomes super sensitive. Very easy to overfit either to track2 or track1. Eg. passing on track2 for 30mph but failed with Zigzag on black bridge on track1.
  - steps:
    git checkout model.h5 model.json
    p3 model.py track2_recover_2.h5.ori_shadow 3
    (following is another git check in)
    p3 model.py data.h5.shadow_lr 1 (track1) - baby step
    p3 model.py track2_recover.h5.ori_shadow 1 - baby step
    p3 model.py track2_recover_2.h5.ori_shadow 1 - baby step

