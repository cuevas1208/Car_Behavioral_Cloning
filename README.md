# Project: Car Behavioral Cloning

Overview
---
This project, deep neural networks and convolutional neural networks tools are used to clone driving behavior.
The model is trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the track in the simulator.

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
trains the data seti

### model.py
contains the model architecture

## load_dataset.py
Splits dataset into validation and training sets 

## preprocess_augmentation.py
Creates the batch data to be use for training, aswell as augmented brightness, and flip image in the left/right direction.

