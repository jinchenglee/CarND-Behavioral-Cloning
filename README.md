
# **Behavioral Cloning Project** 

---

The goals / steps of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report


![alt text][image10]

Above animation shows a scene from track2 that how the trained neural network determines a (somewhat) sharp right turn, the color overlay on the image shows where in the image made the model decide this. Numbers show the turning angle. 


[//]: # (Image References)
[image1]: ./md_images/model_horizontal.png "Model Visualization"
[image2]: ./md_images/GAM1_track2.jpg "Gradient Activation Mapping"
[image3]: ./md_images/nvidia_end2end_net.png "NVidia end-to-end driving net"
[image4]: ./md_images/at_lane_center_track1.jpg "At lane center example image"
[image5]: ./md_images/recover_to_center_track2.jpg "Recovery example Image"
[image6]: ./md_images/aug_image_brightness.png "Imge brightness augmentation"
[image7]: ./md_images/aug_image_shadow.png "Image shadow augmentation"
[image8]: ./md_images/quiver_screenshot.png "Screenshot of quiver page to see convolution layer internals"
[image9]: ./md_images/grad_cam.png "Grad-CAM diagram"
[image10]: ./md_images/cam.gif "Gif annimation of Grad-CAM of conv layer4 output"
[image11]: ./md_images/bad_data.png "Bad training data example"
[image12]: ./md_images/udacity_data_analysis.png "Udacity training data analysis"

[//]: # (blog or webpages references)
[link1]: https://jacobgil.github.io/deeplearning/vehicle-steering-angle-visualizations "Blog: Vehicle steering angle visualization"
[link2]: https://arxiv.org/pdf/1512.04150.pdf "Paper: Learning Deep Features for Discriminative Localization"
[link3]: https://arxiv.org/pdf/1610.02391v1.pdf "Paper: Grad-CAM. Visual Explanations from Deep Networks via Gradient-based Localization"
[link4]: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "NVidia end-to-end neural network paper"
[link5]: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.2d9nkoc46 "Vivek's blog on image augmentation"
[link6]: https://github.com/keplr-io/quiver "Quiver engine page"
[Udacity self driving car simulator]: https://github.com/udacity/self-driving-car-sim

### Quick explanation and HOWTOs

#### 1. Files explanation.

My submitted project includes the following files:
- model.py contains the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network 
- track1\_recording.mp4, track2\_recording.mp4 recorded the process how the model passed both tracks. 

Other files in the repository:
-  README.md, this file (earlier versions recorded the history of baby steps in training and validation the model).
- analyze_data.py is used to analyze the distribution of recorded training data, say histogram of steering angle against speed or throttle.
- cam.py is used to mapping a set of (recorded, but not necessaily) images to (GAM) graidents activation mappings, which is used to help understand what piece of the image the model sees determines its steering decision. 
- fitgen_test.py is used to dump datagenerator generated images. Used for sanity checking whether the changes are desired. 
- preprocess.py is used to turn recorded images and angles into HDF5 data files, along with optional data augmentations, say image brightness change, left/right image augmentation, random shadow generation etc. 
- quiver\_test.py is used to leverage quiver\_engine library to show neural network internals of all conv layers.

#### 2. How to run
Using the Udacity provided simulator (earlier one, track 2 was curvy dark road in black mountains) and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python3 drive.py model.h5
```

#### 3. How to train

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
```sh
python3 model.py <data.h5> <epoch_cnt>
```

The data.h5 file is prepared by preprocess.py in HDF5 format using data augmentation techniques described below. Raw data is recorded in a directory using [Udacity self driving car simulator]. To generate data.h5 file:
```sh
python3 preprocess.py <data_dir> [flip]
```

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model started with NVidia end-to-end neural network described in this paper ([link4]) and it was slightly modified. 

It consists of a convolution neural network with 5 layers of convolution of 3x3 or 5x5 filter sizes and depths between 36 and 95 (model.py lines 71-113). Convolution layers were widened during the fine-tuning process. The architecture of my network is shown in a below section. 

![alt text][image3]

The model includes RELU layers to introduce nonlinearity at the output of every layer, and the data is normalized in preprocess.py (line 280) by a manually written normalization function normalize_color().

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py code 81, 85, 89, 93, 99, 103). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py code line 47)for current datasets collected in two simulator tracks. The model was tested by running it through the simulator and ensuring that the vehicle could stay on both the tracks.

 (Update: 2017 May. The trained NN turned out to badly overfitting when trying on yet another new track in the newer version of car simulator. This is understood as the training dataset is so limited, also the back-annotated heatmap shows the trained NN doens't look at the desired features, such as lanemarks.)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 123). 

Other optimizers were tried during the training process, including SGD and RMSprop, but it ended up with adam. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try, error, analyze failures and improve. 

NVidia-model: My first step was to use a convolution neural network model similar to the NVidia end-to-end paper. I thought this model might be appropriate because it is a proven model that works for real-world road autonomous driving. 

Image cropping: I started with dataset provided by Udacity only. Also, the nvidia network expects image input size of 200x66, and because I believe the upper 1/3-1/4 part of the input image has no meaning to determine my steering angle, I did a cropping of the upper part then scale to the size of 200x66 in preprocessing.py. 
``` python
            img = cv2.imread(DATA_DIR+"/"+row[0])
            img_crop = img[56:160,:,:]
            img_resize = cv2.resize(img_crop, (200,66))
```
Started simple: The training process started with only three images, one with right steering angle, one with left steering angle and the last one with almost 0 angle. I verified my initial Keras model coding can overfit with these three images as input. This serves as a very good practice of sanity check that the model is crafted right and has learning capability.

Train/Validate split: In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

Overfitting: To combat the overfitting, I modified the model by inserting dropouts to layers so that each layer can learn "redundant" features that even some are dropped in dropout layer, it can still predict the right angle. It did work. 

Test: The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (say, the first left turn before black bridge, the left turn after black bridge, and then the right turn after that)... to improve the driving behavior in these cases, I purposely recorded recovery behavior (from curb side to center of the road) along the tracks. Then the car can finish track 1 completely. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Training data selection and preparation is a key starting step to pave base for all future work. Otherwise, it is just garbage in garbage out -- waste of time. 

When I casually collected my data, my data looks like this: left turn dominant and speed capped at fastest pace. Such data won't cover all/enough cases for model to learn good behavior in all scenarios in simulator or real world. 

![alt text][image11]

Udacity provided training data contains good driving behavior. 

![alt text][image12]

I started with Udacity's data using center image only. Here is an example image of center lane driving:

![alt text][image4]

I also applied a scaling factor to left/right image as below.

```python
		SMALL_SCALE = 0.9
		LARGE_SCALE = 1.1

		# Scale angle a bit for left/right images
		if angle > 0: # right turn
			l_angle = LARGE_SCALE * angle
			r_angle = SMALL_SCALE * angle
		else: # left turn
			l_angle = SMALL_SCALE * angle
			r_angle = LARGE_SCALE* angle
```

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image5]


Then I repeated this process on track two in order to get more data points.

After the collection process, I had ~20,000 number of data points. I then preprocessed this data by data augmentation, For example, modifying image brightness histogram and adding random shadows. For example, here are example images:

![alt text][image6]
Brightness augmentation

![alt text][image7]
Random shadow augmentation (copied code from Vivek's blog Vivek's blog  [link5]), which helps A LOT for track 2 with shadows on. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4-5 as evidenced by validation loss no longer goes down. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Interpretation of the model 
During the process of training, I felt very uneasy as it is almost like a blackbox. Whenever the model failed to proceed at a certain spot, it is very hard to tell what went wrong. Although my model passed both tracks, the process of try and error and meddle around with different combination of configurations is quite frustrating. 

### 1. Quiver engine
Quiver engine (github: [link6]) is a web-based tool based on Java and Python/Keras to grab internal info. about your neural network. 

Below is an example page showing the activation output of my model's first convolution layer. 

![alt text][image8]

### 2. Gradient (Class) Activation Mapping
The quiver engine is helpful, but not very straight-forward, as there are too many filters at each convolution layer. At the end of my project, I found a very good blog ([link1]) describing the idea of Activation Mapping. The blog itself was referring to papers: [link2] and [link3]. 

The whole idea is to using heatmap to highlight locality areas contributing most to the final decision. It was designed for classification purpose, but with slight change, it can be applied to our steering angle predictions. 
![alt text][image9]

Cam.py has the implementation. I ran out of time to carefully verifying it, but it seems working pretty well. Here's an example showing the output of convolution layer 4. (Layer 5 is of side 1x18x96 of my design, which lost 1 dimension so conv4 output is more appropriate. )

![alt text][image2]
Above image shows a scene from track2 that the number 0.281566 indicates a (somewhat) sharp right turn, the color overlay on the image shows where in the image made the model decided this.

![alt text][image10]

## Todo:
- The model doesn't generalize enough. Need to leverage GAM information to investigate if time allows. 
- To derive from keras datagenerator as my own class so data augmentation can be done on the fly instead of using HDF5 tables now. 
