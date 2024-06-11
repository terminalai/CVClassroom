import sys 
sys.path.append("../CVClassroom")

import pandas as pd
import numpy as np
import scipy.io 
import math 

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


# sampler for validation 
def default_valid_sampler(indices): # NOTE THAT THIS MUST RETURN THEM IN SORTED ORDER 
    return indices[math.floor(len(indices)*0.8):] 


class StanfordCarsDataloader(keras.utils.PyDataset):

    # initialize dataframes 
    trainDF = pd.read_csv('stanford_cars_dataset/train.csv')
    testDF = pd.read_csv('stanford_cars_dataset/test.csv')

    labelDF = pd.read_csv('stanford_cars_dataset/stanford_cars_labels.csv') # NOTE THAT THIS IS 1-INDEXED SO WE NEED TO SUBTRACT 1  

    # since testDF isn't complete, need to add more stuff 
    testDF['Class'] = [(testLabels['annotations'][0][i][-2][0][0]-1) for i in range(len(testDF)) ]


    # path prefixes
    train_path_prefix = 'stanford_cars_dataset/cars_train/cars_train/' 
    test_path_prefix = 'stanford_cars_dataset/cars_test/cars_test/'

    # other stanford cars specific stuff
    n_classes = 196 
    n_broad_labels = 16 
    ns_sub_labels = [14, 10, 14, 13, 14, 14, 15, 12, 15, 12, 11, 11, 11, 11, 9, 10] 

    def __init__(self, mode="train", batch_size=20, data_shape=(240,360,3), 
                 augment=True, shuffle=True, 
                 valid_sampler=default_valid_sampler,
                 finegrained=True, **kwargs): 
        super().__init__(**kwargs)

        assert ((mode=="train") or (mode == 'valid') or (mode == 'test')), 'That data loading mode is not available.' 
        self.mode = mode 
        self.batch_size = batch_size 
        self.data_shape = data_shape 
        self.augment = augment 
        self.shuffle = shuffle 
        self.valid_sampler = valid_sampler
        self.finegrained = finegrained 

        # set indices 
        if (self.mode == 'test'): 
            self.indices = np.arange(StanfordCarsDataloader.testDF.shape[0])  
        elif (self.mode == 'valid'): 
            self.indices = self.valid_sampler(range(StanfordCarsDataloader.trainDF.shape[0]))
        else: 
            valid_indices = self.valid_sampler(range(StanfordCarsDataloader.trainDF.shape[0])) 
            #print("VALID INDICES:",valid_indices)
            self.indices = [] 
            pos = 0 
            for i in range(StanfordCarsDataloader.trainDF.shape[0]): 
                if (pos < len(valid_indices)) and (i == valid_indices[pos]): 
                    pos += 1 
                else: 
                    self.indices.append(i) 


        self.on_epoch_end() # as initialization 


    def on_epoch_end(self): 
        if self.shuffle: 
            np.random.shuffle(self.indices) 
    

    def generate_data(self, indices): # generates data given some indices, not the entire thing 
        x = np.empty((self.batch_size, *self.data_shape)) 
        if self.finegrained:
            y1 = np.empty(self.batch_size, dtype=int) # broad label 
            y2 = np.empty(self.batch_size, dtype=int) # sublabel 
        else:
            y = np.empty(self.batch_size, dtype=int) # labels 

        for i, index in enumerate(indices): # NOTE: NOT self.indices BUT THE PARAMETER 
            if self.mode == "train": 
                x[i,] = img_to_array(load_img(
                    StanfordCarsDataloader.train_path_prefix +
                    StanfordCarsDataloader.trainDF.image[i],
                    target_size=self.data_shape, interpolation='bilinear'))

                if self.finegrained:
                    y1[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.trainDF.Class[i]]['broad_label'] -1 
                    y2[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.trainDF.Class[i]]['sub_label'] -1 
                else:
                    y[i] = StanfordCarsDataloader.trainDF.Class[i] 
            
            elif self.mode == "valid": # take from train path too 
                x[i,] = img_to_array(load_img(
                    StanfordCarsDataloader.train_path_prefix +
                    StanfordCarsDataloader.trainDF.image[i],
                    target_size=self.data_shape, interpolation='bilinear'))

                if self.finegrained:
                    y1[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.trainDF.Class[i]]['broad_label'] -1 
                    y2[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.trainDF.Class[i]]['sub_label'] -1 
                else:
                    y[i] = StanfordCarsDataloader.trainDF.Class[i] 

            elif self.mode == "test": 
                x[i,] = img_to_array(load_img(
                    StanfordCarsDataloader.test_path_prefix +
                    StanfordCarsDataloader.testDF.image[i],
                    target_size=self.data_shape, interpolation='bilinear'))
                
                if self.finegrained:
                    y1[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.testDF.Class[i]]['broad_label'] -1 
                    y2[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.testDF.Class[i]]['sub_label'] -1 
                else:
                    y[i] = StanfordCarsDataloader.testDF.Class[i] 
        
        if self.augment: 
            x = rand_augment(x) 

        if self.finegrained:
            return x, (keras.utils.to_categorical(y1, num_classes = StanfordCarsDataloader.n_broad_labels), 
                       keras.utils.to_categorical(y2, num_classes = max(StanfordCarsDataloader.ns_sub_labels))) 
        else:
            return x, keras.utils.to_categorical(y, num_classes = StanfordCarsDataloader.n_classes) 


    def __len__(self): 
        return int(np.floor(len(self.indices) / self.batch_size))
        '''if self.mode == 'train': 
            return int(np.floor(StanfordCarsDataloader.trainDF.shape[0] / self.batch_size)) 
        else: 
            return int(np.floor(StanfordCarsDataloader.testDF.shape[0] / self.batch_size)) '''
        

    def __getitem__(self, index): # get the batch of data 
        indices = self.indices[index * self.batch_size : (index+1) * self.batch_size] 

        return self.generate_data(indices) 


