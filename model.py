## File: model.py
## Name: Manuel Cuevas
## Date: 02/14/2017
## Project: CarND - Behavioral Cloning
## Desc: convolution neural network model to learn a track
## Ref: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
#######################################################################################
'''
This model was based on the Nvidia model with modification to better fit our needs
This model uses 3 convolutional layers with filter size 7x7, 1x1, 3X3 followed by a
elu activations.
Input:  image size -> 160, 320, 3
Return: logits'''
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Lambda, ELU

class modelClass:
    
    def __init__(self):
        self.model = Sequential()
        self.get_model()
        print("hey")
        
    
    def get_model(self):
        # Preprocess incoming data
        # Normalize the features using Min-Max scaling centered around zero and reshape
        self.model.add(Lambda(lambda x: (x/125.5) - 1., input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
        # Image is crop to left only the important features of the image
        self.model.add(Lambda(lambda x: x[ :, 65:140, 0:320, :], output_shape = (75, 320, 3)))
        print (self.model.output_shape)

        #(None, 75, 320, 3)
        self.model.add(Convolution2D(64, 7, 7, subsample=(2,2), border_mode='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(ELU())
        print (self.model.output_shape)         

        #(None, 19, 80, 64)
        self.model.add(Convolution2D(64, 1, 1, subsample=(2, 2), border_mode="same", activation='relu'))
        self.model.add(Convolution2D(193, 3, 3, subsample=(2, 2), border_mode="same", activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(1,2)))
        self.model.add(ELU())
        print (self.model.output_shape)         

        #(None, 4, 10, 193)
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=None))
        print (self.model.output_shape)         

        #(None, 2, 5, 193)
        self.model.add(Flatten())
        print (self.model.output_shape)

        #(None, 1930)
        self.model.add(Dense(1500))
        self.model.add(Dropout(.2))
        self.model.add(ELU())
        self.model.add(Dense(500))
        self.model.add(Dropout(.2))
        self.model.add(ELU())
        self.model.add(Dense(1))
        # compile self.model using adam
        self.model.compile(optimizer="adam", loss="mse")

    '''
    Stores the trained model to json format
    Input:  model - trained model
            location, - folder location
    Return: void '''
    def savemodel(self, location = "./steering_model"):
        import os
        import json
        print("Saving model weights and configuration file.")

        if not os.path.exists(location):
            os.makedirs(location)

        # serialize model to JSON and weights to h5
        self.model.save_weights(location + "/model.h5", True)
        with open(location +'/model.json', 'w') as outfile:
            outfile.write(self.model.to_json())
            
        print("Saved model to disk")






