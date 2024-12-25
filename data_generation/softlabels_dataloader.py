# IMPORTS
import sys
sys.path.append("../CVClassroom")

import pandas as pd
import numpy as np
import keras.utils

# VARIABLES
seed = 12; np.random.seed(seed)

# FUNCTIONS
def default_vsampler(indices): return indices[:int(len(indices)*0.8)]

# DATALOADER
class SoftlabelsDataloader(keras.utils.PyDataset):
    

    def __init__(self, mode, data_folder:str, labels_file:str, expert_class, use_gating_mdl=False, batch_size=20, data_shape=(224, 224, 3), mod_functions:list=[], do_not_repeat_train_idx_in_valid = False, valid_sampler=default_vsampler, shuffle=True,**kwargs): 
        
        # initialise object attributes
        self.dtf = data_folder
        self.bs = batch_size
        self.dts = data_shape
        self.modfuncs = mod_functions # input/output format: PIL Image
        self.shuffle = shuffle
        
        # initialise PyDataset
        super().__init__(**kwargs)

        # fetch labels
        self.labelDF = pd.read_csv(labels_file)

        # check validity of mode
        assert ((mode == "train") or (mode == 'valid') or (mode == 'test')), 'That data loading mode is not available.'

        # filter data based on mode
        self.indices = self.labelDF.index.to_list
        
        # shuffle data
        if self.shuffle:
            np.random.shuffle(self.indices)

        if mode == "train" and do_not_repeat_train_idx_in_valid: 
            temp = valid_sampler(self.indices)
            self.indices = list(filter(lambda item: item not in temp, self.indices))
            del temp
        elif mode == "valid":
            self.indices = valid_sampler(self.indices)
        
        # filter data based on expert class and convert to numpy array
        self.indices = np.array(list(filter(lambda x: expert_class == self.labelDF.iloc(x)["broad_label"], self.indices)))


    def on_epoch_end(self): 
        if self.shuffle:
            np.random.shuffle(self.indices)


    def create_batch(self, indices): # generates data given some indices, not the entire thing 
        # generate numpy arrays for the images and labels
        x = np.empty((self.bs, *self.dts)) # images
        y = np.empty(self.bs, dtype=int) # labels 

        # convert image to array and fill numpy arrays with data and labels
        for i in indices:
            # get image
            img = keras.utils.load_img(self.dtf + self.labelDF.iloc(i)["path"][1:], target_size=self.dts, interpolation='bilinear') # [1:] is to remove the dot at the start of the filename string
            
            # modify the image
            if len(self.modfuncs):
                for func in self.modfuncs:
                    img = func(img)
            
            # add image to created batch
            x[i,] = keras.utils.img_to_array(img)
            y[i] = self.labelDF.iloc(i)["sub_label"]
            
        # return created batch
        return x, y


    def __len__(self): 
        return int(len(self.indices) / self.bs)


    def __getitem__(self, index): # get the batch of data 
        indices = self.indices[index * self.bs : (index+1) * self.bs] 

        return self.create_batch(indices) 


