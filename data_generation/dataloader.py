import keras 
from keras.datasets import cifar100 
from keras_cv.layers import RandAugment 

import numpy as np 

from misc import params 

rand_augment = RandAugment([0,255], **params.augment_params) # define augmentation function 

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine") # get data from some source 
assert x_train.shape == (50000, 32, 32, 3) 
assert x_test.shape == (10000, 32, 32, 3) 
assert y_train.shape == (50000, 1) 
assert y_test.shape == (10000, 1) 


class Dataloader(keras.utils.Sequence): 
    def __init__(self, mode="train", batch_size=20, data_shape=(32,32,3), n_classes=100, augment=True, shuffle=True): 
        self.mode = mode 
        self.batch_size = batch_size 
        self.data_shape = data_shape 
        self.n_casses = n_classes 
        self.augment = augment 
        self.shuffle = shuffle 
        self.on_epoch_end() # as initialization 
    
    def on_epoch_end(self): 
        self.indices = np.arange(len(self.list_IDs)) 
        if self.shuffle: 
            np.random.shuffle(self.indices) 
    
    def generate_data(self, indices): 
        x = np.empty((self.batch_size, *self.data_shape)) 
        y = np.empty(self.batch_size, dtype=int) # labels 

        for i, index in enumerate(indices): # NOTE: NOT self.indices BUT THE PARAMETER 
            if self.mode == "train": 
                x[i,] = x_train[index] 
                y[i] = y_train[index] 
            else: 
                x[i,] = x_test[index] 
                y[i] = y_test[index] 
        
        if self.augment: 
            x = rand_augment(x) 

        return x, keras.utils.to_categorical(y, num_clases=self.n_classes) 

    def __len__(self): 
        if self.mode == 'train': 
            return int(np.floor(x_train.shape[0] / self.batch_size)) 
        else: 
            return int(np.floor(x_test.shape[0] / self.batch_size)) 
        
    def __getitem__(self, index): # get the batch of data 
        indices = self.indices[index * self.batch_size : (index+1) * self.batch_size] 

        return self.generate_data(indices) 


