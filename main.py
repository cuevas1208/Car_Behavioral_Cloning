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
from load_dataset import LoadData
from model import modelClass
from preprocess_augmentation import *

if __name__ == '__main__':
    #Parameters
    left_stcor = 0.28
    right_stcor = -0.28
    imgN = 5 
    
    #Load Data img and CSV
    dataO = LoadData(location = '../data/driving_log.csv', validationSplit = .2)

    print('data', type(dataO.data))
    print('valid_data', type(dataO.valid_data))
    print('train_data', type(dataO.train_data))

    #Split labels
    y_train, y_validation = dataO.split_labels(imgN = 5, left_stcor = 0.28, right_stcor = -0.28)

    #Get model
    modelO = modelClass()

    #Prepare data batch
    batchSize = 120
    sampPerEpoch = (len(dataO.train_data)*imgN)-((len(dataO.train_data)*imgN)%batchSize)
    valSamp = (len(dataO.valid_data)*imgN)-((len(dataO.valid_data)*imgN)%batchSize)

    #train the model using the generator function
    modelO.model.fit_generator(genImage(sampPerEpoch, batchSize, dataO.train_data, y_train, imgN),
                               validation_data = genImage(valSamp, batchSize, dataO.valid_data, y_validation, imgN),
                               nb_val_samples=valSamp, samples_per_epoch = sampPerEpoch, nb_epoch=5, verbose = 1)##Saving the model

    #Save model
    modelO.savemodel(location = "./steering_model")
