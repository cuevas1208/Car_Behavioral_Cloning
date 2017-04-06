## File: main.py
## Name: Manuel Cuevas
## Date: 2/08/2016
## Project: CarND - Behavioral Cloning 
## Desc: Load, Preprocces, trains and saves car simulation images.
## Usage: This project identifies a series of steps that identify stering wheel
## angle from images.
## Tools learned in Udacity CarND program were used to identify lane lines on the road.
#######################################################################################
#importing useful packages
import pickle
import numpy as np
import math
import tensorflow as tf
import unicodecsv
import random
import matplotlib.image as mpimg
import numpy as np
tf.python.control_flow_ops = tf  #Fix error with TF and Keras

from preprocess_augmentation import *
from model import *
from load_dataset import *

#Parameters
left_stcor = 0.28
right_stcor = -0.28
imgN = 5

#Load Data img and CSV
data, valid_data, train_data = load_datasets(location = '../data/driving_log.csv', validationSplit = .2)

#Split labels
y_train, y_validation = split_labels(train_data, valid_data, imgN = 5,
                                     left_stcor = 0.28, right_stcor = -0.28)

#Get model
model = get_model()

#Prepare data batch
batchSize = 120
sampPerEpoch = (len(train_data)*imgN)-((len(train_data)*imgN)%batchSize)
valSamp = (len(valid_data)*imgN)-((len(valid_data)*imgN)%batchSize)

#train the model using the generator function
model.fit_generator(genImage(sampPerEpoch, batchSize, train_data, y_train, imgN), validation_data = genImage(valSamp, batchSize, valid_data, y_validation, imgN), nb_val_samples=valSamp, samples_per_epoch = sampPerEpoch, nb_epoch=5, verbose = 1)##Saving the model

#Save model
savemodel(model, location = "./steering_model")
