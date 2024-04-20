import sys 
sys.path.append("../CVClassroom")

import pandas as pd
import numpy as np
import scipy.io 

import keras.utils 
from keras.preprocessing.image import load_img, img_to_array
from keras_cv.layers import RandAugment 

import params 

seed = 10
np.random.seed(seed) 



# define augmentation function 
rand_augment = RandAugment([0,255], **params.augment_params) 


# prepare this bcs need to preload 
testLabels = scipy.io.loadmat('stanford_cars_dataset/cars_test_annos_withlabels (1).mat') 


class StanfordCarsDataloader(keras.utils.Sequence):

    # initialize dataframes 
    trainDF = pd.read_csv('stanford_cars_dataset/train.csv')
    testDF = pd.read_csv('stanford_cars_dataset/test.csv')

    # since testDF isn't complete, need to add more stuff 
    testDF['Class'] = [(testLabels['annotations'][0][i][-2][0][0]-1) for i in range(len(testDF)) ]


    # path prefixes
    train_path_prefix = 'stanford_cars_dataset/cars_train/cars_train/' 
    test_path_prefix = 'stanford_cars_dataset/cars_test/cars_test/'

    # other stanford cars specific stuff
    n_classes = 196 

    def __init__(self, mode="train", batch_size=20, data_shape=(240,360,3), augment=True, shuffle=True): 
        assert ((mode=="train") or (mode == 'test')), 'That data loading mode is not available.' 
        self.mode = mode 
        self.batch_size = batch_size 
        self.data_shape = data_shape 
        self.augment = augment 
        self.shuffle = shuffle 
        self.on_epoch_end() # as initialization 
    

    def on_epoch_end(self): 
        self.indices = np.arange(StanfordCarsDataloader.trainDF.shape[0]) 
        if self.shuffle: 
            np.random.shuffle(self.indices) 
    

    def generate_data(self, indices): # generates data given some indices, not the entire thing 
        x = np.empty((self.batch_size, *self.data_shape)) 
        y = np.empty(self.batch_size, dtype=int) # labels 

        for i, index in enumerate(indices): # NOTE: NOT self.indices BUT THE PARAMETER 
            if self.mode == "train": 
                x[i,] = img_to_array(load_img(
                    StanfordCarsDataloader.train_path_prefix +
                    StanfordCarsDataloader.trainDF.image[i],
                    target_size=self.data_shape, interpolation='bilinear'))

                y[i] = StanfordCarsDataloader.trainDF.Class[i] 

            elif self.mode == "test": 
                x[i,] = img_to_array(load_img(
                    StanfordCarsDataloader.test_path_prefix +
                    StanfordCarsDataloader.testDF.image[i],
                    target_size=self.data_shape, interpolation='bilinear'))
                
                y[i] = StanfordCarsDataloader.testDF.Class[i] 
        
        if self.augment: 
            x = rand_augment(x) 

        return x, keras.utils.to_categorical(y, num_classes = StanfordCarsDataloader.n_classes) 


    def __len__(self): 
        if self.mode == 'train': 
            return int(np.floor(StanfordCarsDataloader.trainDF.shape[0] / self.batch_size)) 
        else: 
            return int(np.floor(StanfordCarsDataloader.testDF.shape[0] / self.batch_size)) 
        

    def __getitem__(self, index): # get the batch of data 
        indices = self.indices[index * self.batch_size : (index+1) * self.batch_size] 

        return self.generate_data(indices) 


