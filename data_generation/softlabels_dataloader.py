# IMPORTS
import sys

sys.path.append("../CVClassroom")

import pandas as pd
import numpy as np
import scipy.io 
import math 

import keras.utils # type: ignore

# VARIABLES
seed = 10
np.random.seed(seed)

# FUNCTIONS
def default_vsampler(indices): return indices[:math.floor(len(indices)*0.8)]

# DATALOADER
class SoftlabelsDataloader(keras.utils.PyDataset):
                                                                                                                                         # input is all indices, must give indices for validation as output
    def __init__(self, mode, data_folder:str, labels_file:str, expert_class, do_not_repeat_train_idx_in_valid = False, batch_size=20, data_shape=(224, 224, 3), modify_data=False, mod_functions:list=None, valid_sampler=default_vsampler, shuffle=True,**kwargs): 
        
        # initialise object variables
        self.dtf = data_folder
        self.bs = batch_size
        self.dts = data_shape
        self.aug = modify_data
        self.augfuncs = mod_functions # input/output format: PIL Image
        self.shuffle = shuffle
        
        # initialise PyDataset
        super().__init__(**kwargs)

        # fetch labels
        self.labelDF = pd.read_csv(labels_file)

        # check validity of mode
        assert ((mode == "train") or (mode == 'valid') or (mode == 'test')), 'That data loading mode is not available.'

        # filter data based on mode
        if mode == "test": self.indices = self.labelDF.index.to_list
        elif mode == "train": 
            if do_not_repeat_train_idx_in_valid:
                allidx = self.labelDF.index.to_list; temp = valid_sampler(allidx); self.indices = list(filter(lambda item: item not in temp, allidx)); del allidx, temp
            else:
                self.indices = self.labelDF.index.to_list
        else:
            self.indices = valid_sampler(self.labelDF.index.to_list)
        
        # filter data based on expert class and convert to numpy array
        self.indices = np.array(list(filter(lambda x: expert_class == self.labelDF.iloc(x)["broad_label"], self.indices)))

        # shuffle data
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self): 
        if self.shuffle:
            np.random.shuffle(self.indices)
    

    def create_batch(self, indices): # generates data given some indices, not the entire thing 
        # generate numpy arrays for the images and labels
        x = np.empty((self.bs, *self.dts))
        y = np.empty(self.bs, dtype=int) # labels 

        # convert image to array and fill numpy arrays with data and labels
        for i in indices: # NOTE: NOT self.indices BUT THE PARAMETER 
            
            img = keras.utils.load_img(
                self.dtf +
                self.labelDF.iloc(i)["path"][1:], # [1:] is to remove the dot at the start of the path file string
                target_size=self.dts, interpolation='bilinear')
            
            # augment the images
            if self.aug:
                for func in self.augfuncs:
                    img = func(img)

            # i think i saw inconsistency in the slashes in the filename strings
            # return created batch

            x[i,] = keras.utils.img_to_array(img)
            y[i] = self.labelDF.iloc(i)["sub_label"]

        return x, y


    def __len__(self): 
        return int(np.floor(len(self.indices) / self.bs))      

    def __getitem__(self, index): # get the batch of data 
        indices = self.indices[index * self.bs : (index+1) * self.bs] 

        return self.create_batch(indices) 


