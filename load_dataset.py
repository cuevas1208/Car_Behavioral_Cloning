## File: preprocess_augmentation.py
## Name: Manuel Cuevas
## Date: 02/14/2017
## Project: CarND - Behavioral Cloning
## Desc: Splits dataset not in batch data. In addition, an implementation
##       of augmented brightness, and flip image in the left/right direction.
## Usage:Data augmentation allows the network to learn the important features
##      that are invariant for the object classes. In addition, augmentation is
##      used to artificially increase the size of the dataset.
#######################################################################################
#importing useful packages
import numpy as np
import matplotlib.image as mpimg
from skimage import exposure
import unicodecsv
import random

'''Load Data img and CSV
Splits and randomize dataset
Input:  Location - folder location to load dataset
        validationSplit - decimal portion to be taken for validation
Return: data - original data set
        valid_data- validation data set
        train_data- train data set '''
def load_datasets(location = '../data/driving_log.csv', validationSplit = .2):
    with open('../data/driving_log.csv', 'rb') as f:
        reader = unicodecsv.DictReader(f)
        data = list(reader)

    #shufle data
    random.seed(4)
    random.shuffle(data)

    #split data

    validationLen = int(len(data)*.2)
    valid_data = data[:validationLen]
    train_data = data[validationLen:]

    assert np.array_equal((len(train_data) + len(valid_data)), len(data)), 'data is incomplete'
    print('Dataset loaded.')
    print (mpimg.imread("../data/"+train_data[0]['center']).shape)
    
    return (data, valid_data, train_data)

'''
appendLabels
Creates a list of labels  
Input:  data - dictionary of data set
        label_List - array to store the list
        left_stcor, right_stcor - left and right steering wheel offset
Return: NONE'''
def appendLabels(data, label_List, left_stcor, right_stcor):
    for i in range(0,len(data)):
        label_List.append(float(data[i]['steering']))
        label_List.append(float(data[i]['steering'])+left_stcor)
        label_List.append((float(data[i]['steering'])+left_stcor)*-1)
        label_List.append(float(data[i]['steering'])+right_stcor)
        label_List.append((float(data[i]['steering'])+right_stcor)*-1)

'''
split_labels
Thickness used random rotations, Zoom, Brightness, shear, translation, color tones, .
Input:  train_data, valid_data, - Train and validation dataset
        imgN- number of images per frame
        left_stcor, right_stcor - left and right steering wheel offset
Return: train and valid images labels arrays'''
def split_labels(train_data, valid_data, imgN, left_stcor = 0.28, right_stcor = -0.28):
    a = []
    b = []
    appendLabels(train_data, a, left_stcor, right_stcor)
    appendLabels(valid_data, b, left_stcor, right_stcor)
    assert (len(train_data)*imgN  == len(a)), 'labels length is not equal to images length'
    assert (len(valid_data)*imgN  == len(b)), 'labels length is not equal to images length'

    #validate_dataSet()
    return (np.array(a), np.array(b))
'''
validate_dataSet
Validate images and labels, veify data collected match their labels
Input:  NONE
Return: NONE'''
def validate_dataSet():
    index = index - index%imgN
    print ((data[int(index/imgN)]['steering']).replace(" ", ""))
    print (y[index])
    assert float((data[int(index/imgN)]['steering']).replace(" ", ""))  == y[index]
    index = random.randint(0, len(y))
    index = index - index%imgN
    assert float((data[int(index/imgN)]['steering']).replace(" ", ""))  == y[index]
    assert float((data[int(index/imgN)]['steering']).replace(" ", ""))+left_stcor  == y[index+1]
    assert (float((data[int(index/imgN)]['steering']).replace(" ", ""))+left_stcor)*-1  == y[index+2]
    assert float((data[int(index/imgN)]['steering']).replace(" ", ""))+right_stcor  == y[index+3]
    assert (float((data[int(index/imgN)]['steering']).replace(" ", ""))+right_stcor)*-1  == y[index+4]
    return




