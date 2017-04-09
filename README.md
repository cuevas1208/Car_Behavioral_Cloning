# Project: Car Behavioral Cloning

Overview
---
This project, deep neural networks and convolutional neural networks tools are used to clone driving behavior.
The model is trained, validated and tested using Keras API. The model will output a steering angle to an autonomous vehicle.
##### For more information about this project visit the [Wiki page](https://github.com/cuevas1208/Car_Behavioral_Cloning/wiki)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the track in the simulator.

Get started
--
#### Train your model
Download project repository. Open Git Bash and type
```sh
git clone main.py https://github.com/cuevas1208/Car_Behavioral_Cloning.git
```
Open terminal on the project and use this command to train your model:
```sh
python main.py 
```
When training is completed you should have two files under the `steering_model` folder `model.json` and `model.h5`

#### Simulation  

Download the simulation from this [link](https://github.com/udacity/self-driving-car-sim)

Use `drive.py` to run your model.`drive.py` requires you to have `model.h5` and `model.json` file saved under 'steering_model' folder

Once the model has been saved, it can be used with drive.py using this command:
```sh
python drive.py 
```

Open your simulation and run your simulation mode. The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is a known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

Dataset
---
The simulator will use the steering angles outs from the model to drive a car around a track.
The simulator has a setting to record data to train the model. The data set consists of image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

Dependencies
---
This project requires **Python 3.5** and the following Python libraries installed:
- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [Keras](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)


## Details About Files In This Directory

### drive.py
runs model with the simulator 
                                                                                                                          
### main.py
trains the data set

### model.py
contains the model architecture

### load_dataset.py
Splits dataset into validation and training sets 

### preprocess_augmentation.py
Creates the batch data to be used for training, as well as augmented brightness, and flip image in the left/right direction.
