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

class LoadData:
    
    data = []
    valid_data = []
    train_data = []
    
    def __init__(self, location = '../data/driving_log.csv', validationSplit = .2):
        self.location = location
        self.validationSplit = validationSplit
        self.load_datasets()

    '''Load Data img and CSV
    Splits and randomize dataset
    Input:  Location - folder location to load dataset
            validationSplit - decimal portion to be taken for validation
    Return: data - original data set
            valid_data- validation data set
            train_data- train data set '''
    def load_datasets(self):
        with open(self.location, 'rb') as f:
            reader = unicodecsv.DictReader(f)
            self.data = list(reader)

        #shufle data
        random.seed(4)
        random.shuffle(self.data)

        #split data
        validationLen = int(len(self.data)*.2)
        self.valid_data = self.data[:validationLen]
        self.train_data = self.data[validationLen:]

        assert np.array_equal((len(self.train_data) + len(self.valid_data)), len(self.data)), 'data is incomplete'
        print('Dataset loaded.')
        print (mpimg.imread("../data/"+self.train_data[0]['center']).shape)
        
        return

    '''
    split_labels
    Thickness used random rotations, Zoom, Brightness, shear, translation, color tones, .
    Input:  train_data, valid_data, - Train and validation dataset
            imgN- number of images per frame
            left_stcor, right_stcor - left and right steering wheel offset
    Return: train and valid images labels arrays'''
    def split_labels(self, imgN, left_stcor = 0.28, right_stcor = -0.28):
        a = []
        b = []
        self.appendLabels(self.train_data, a, left_stcor, right_stcor)
        self.appendLabels(self.valid_data, b, left_stcor, right_stcor)
        assert (len(self.train_data)*imgN  == len(a)), 'labels length is not equal to images length'
        assert (len(self.valid_data)*imgN  == len(b)), 'labels length is not equal to images length'

        self.validate_dataSet(b, left_stcor, right_stcor, imgN)
        
        return (np.array(a), np.array(b))

    '''
    appendLabels
    Creates a list of labels  
    Input:  data - dictionary of data set
            label_List - array to store the list
            left_stcor, right_stcor - left and right steering wheel offset
    Return: NONE'''
    @staticmethod
    def appendLabels(data, label_List, left_stcor, right_stcor):
        for i in range(0,len(data)):
            label_List.append(float(data[i]['steering']))
            label_List.append(float(data[i]['steering'])+left_stcor)
            label_List.append((float(data[i]['steering'])+left_stcor)*-1)
            label_List.append(float(data[i]['steering'])+right_stcor)
            label_List.append((float(data[i]['steering'])+right_stcor)*-1)
            
    '''
    validate_dataSet
    Validate images and labels, veify data collected match their labels
    Input:  NONE
    Return: NONE'''
    def validate_dataSet(self, y, left_stcor, right_stcor, imgN):
        index = random.randint(0, len(y))
        index = index - index%imgN
        assert  float((self.data[int(index/imgN)]['steering']).replace(" ", ""))                   == y[index]
        assert  float((self.data[int(index/imgN)]['steering']).replace(" ", ""))+left_stcor        == y[index+1]
        assert (float((self.data[int(index/imgN)]['steering']).replace(" ", ""))+left_stcor)*-1    == y[index+2]
        assert  float((self.data[int(index/imgN)]['steering']).replace(" ", ""))+right_stcor       == y[index+3]
        assert (float((self.data[int(index/imgN)]['steering']).replace(" ", ""))+right_stcor)*-1   == y[index+4]
        return




