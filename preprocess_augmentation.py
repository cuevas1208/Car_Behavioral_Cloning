## File: preprocess_augmentation.py
## Name: Manuel Cuevas
## Date: 02/14/2017
## Project: CarND - Behavioral Cloning
## Desc: Creates the batch data to be use for training. Along with ther and implementation
##       of augmented brightness, and flip image in the left/right direction.
## Usage:Data augmentation allows the network to learn the important features
##      that are invariant for the object classes. In addition, augmentation is
##      used to artificially increase the size of the dataset.
#######################################################################################
#importing useful packages
import numpy as np
import matplotlib.image as mpimg
from skimage import exposure

'''loadImg
Incorporates the sides right and left images to the data set,
flip augmented image in the left/right direction is used
Input:  n     - location of the image
        data  - original image dataset
Return: x     - a set of images at a from the same time'''
def loadImg(n, data):
    return np.array([
            (mpimg.imread("../data/"+data[n]['center'])),
            (mpimg.imread(("../data/"+data[n]['left']).replace(" ", ""))) ,
            (np.fliplr(mpimg.imread(("../data/"+data[n]['left']).replace(" ", "")))),
            (mpimg.imread(("../data/"+data[n]['right']).replace(" ", ""))),
            (np.fliplr(mpimg.imread(("../data/"+data[n]['right']).replace(" ", ""))))
            ])

'''imgArray
Creates batch of images from a large data set
Input:  start - location to begin pulling images from
        end   - location to stop pulling images from
        data  - original images array
Return: x     - batch of images pulled from data input'''
def imgArray(start, end, data, imgN):
    start = int(start/imgN)
    end = int(end/imgN)
    
    x = loadImg(start,data)
    for n in range (start+1, end):
        temp = loadImg(n,data)
        x = np.concatenate((x, temp), axis=0)
    return x

'''genImage
Return a batch of images ready for training
Input:  limit, batch, data, val_data
Return: x - Array of training images
        y - array of labels'''
def genImage(limit, batch, data, val_data, imgN):
    batch = batch - batch%imgN
    limit = limit - limit%batch

    while 1:
        for i in range(0, limit, batch):
            x = imgArray(i, (i + batch), data, imgN)
            y = val_data[i : i + batch]
            yield (x, y)

'''brightness
Randomly change the brightness of the image
Input:  n     - location of the image
        data  - original image dataset
Return: x     - a set of images at a from the same time'''
def brightness(img, b_range):
    gamma = np.random.uniform(1-b_range, 1+b_range)
    img = exposure.adjust_gamma(img, gamma, 1)
    return img




