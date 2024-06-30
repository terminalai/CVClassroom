import sys 
sys.path.append("../CVClassroom")

import pandas as pd
import numpy as np
import scipy.io 
import math 

import keras.utils 
from keras.preprocessing.image import load_img, img_to_array
from keras_cv.layers import RandAugment, RandomFlip, RandomRotation 

import params 

seed = 10
np.random.seed(seed) 



# define augmentation function 
default_augment_func = keras.Sequential([ RandAugment([0,255], **params.augment_params), 
                                 RandomFlip(mode='horizontal'), 
                                 RandomRotation(factor=0.1), 
                                 ])


# prepare this bcs need to preload 
testLabels = scipy.io.loadmat('stanford_cars_dataset/cars_test_annos_withlabels (1).mat') 


# sampler for validation 
def default_valid_sampler(indices): # NOTE THAT THIS MUST RETURN THEM IN SORTED ORDER 
    return indices[math.floor(len(indices)*0.8):] 


class StanfordCarsDataloader(keras.utils.PyDataset):

    # initialize dataframes for data 
    trainDF = pd.read_csv('stanford_cars_dataset/train.csv')
    testDF = pd.read_csv('stanford_cars_dataset/test.csv')

    # since testDF isn't complete, need to add more stuff 
    testDF['Class'] = [(testLabels['annotations'][0][i][-2][0][0]-1) for i in range(len(testDF)) ]


    # path prefixes
    train_path_prefix = 'stanford_cars_dataset/cars_train/cars_train/' 
    test_path_prefix = 'stanford_cars_dataset/cars_test/cars_test/'

    # splitting of labels into sublabels 
    n_classes = 196 
    n_broad_labels = 16 
    ns_sub_labels = [14, 10, 14, 13, 14, 14, 15, 12, 15, 12, 11, 11, 11, 11, 9, 10] 
    labelDF = pd.read_csv('stanford_cars_dataset/stanford_cars_labels.csv') # NOTE THAT THIS IS 1-INDEXED 

    @classmethod 
    def labels_for_broad_label(broadlabel): # all these are 1-indexed 
        return list(StanfordCarsDataloader.labelDF[StanfordCarsDataloader.labelDF['broad_label'] == broadlabel]['label']) 
    
    @classmethod 
    def broad_sub_labels_to_label(broad_label, sub_label): 
        return CarConnectionDataloader.labelDF['label'][CarConnectionDataloader.labelDF['broad_label'] == broad_label]['label'][CarConnectionDataloader.labelDF['sub_label'] == sub_label]['label'] 

    label_csv_paths = {
        0: 'stanford_cars_dataset/stanford_cars_labels.csv', 
    }

    @classmethod 
    def switch_labelling_method(new_mtd): 
        csv_path = "" 
        if type(new_mtd) == int: 
            assert new_mtd in StanfordCarsDataloader.label_csv_paths.keys(), "Label method does not exist. " 
            csv_path = StanfordCarsDataloader.label_csv_paths[csv_path] 
        elif type(new_mtd) == str: 
            csv_path = new_mtd 
        else: 
            raise ValueError("new_mtd must be int (using preset labellings) or str (custom label csv path)!") 
        
        # TODO: CHANGE LABELLING METHOD 

    def __init__(self, mode="train", batch_size=20, data_shape=(240,360,3), 
                 augment=True, augment_func=default_augment_func, shuffle=True, 
                 valid_sampler=default_valid_sampler,
                 finegrained=True, force_same_shape_sublabel=None, 
                 broad_label_filter=None, broad_labels_only=False, 
                 **kwargs): 
        super().__init__(**kwargs)

        assert ((mode == "train") or (mode == 'valid') or (mode == 'test')), 'That data loading mode is not available.' 
        self.mode = mode 
        self.batch_size = batch_size 
        self.data_shape = data_shape 
        self.augment = augment 
        self.augment_func = augment_func 
        self.shuffle = shuffle 
        self.valid_sampler = valid_sampler
        self.finegrained = finegrained 
        if (force_same_shape_sublabel is None): 
            self.force_same_shape_sublabel = not (broad_label_filter is None) 
        else: self.force_same_shape_sublabel = force_same_shape_sublabel 
        self.broad_label_filter = broad_label_filter 
        self.broad_labels_only = broad_labels_only 

        # set indices 
        if (self.mode == 'test'): 
            if self.broad_label_filter is not None : 
                self.indices = StanfordCarsDataloader.testDF.index[
                    StanfordCarsDataloader.testDF['Class'] in StanfordCarsDataloader.labels_for_broad_label(self.broad_label_filter)
                    ] 
            else: 
                self.indices = StanfordCarsDataloader.testDF.index 
        else: 
            if self.broad_label_filter is not None : 
                t_indices = StanfordCarsDataloader.trainDF.index[
                    StanfordCarsDataloader.trainDF['Class'] in StanfordCarsDataloader.labels_for_broad_label(self.broad_label_filter)
                    ] 
            else: 
                t_indices = StanfordCarsDataloader.trainDF.index 
            valid_indices = self.valid_sampler(t_indices) 

            #print("VALID INDICES:",valid_indices) 

            if self.mode == 'valid': 
                self.indices = valid_indices
            else: 
                self.indices = [] 
                pos = 0 
                for i in StanfordCarsDataloader.trainDF.index: 
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
            if (not self.broad_labels_only): y2 = np.empty(self.batch_size, dtype=int) # sublabel 
        else:
            y = np.empty(self.batch_size, dtype=int) # labels 

        for i, index in enumerate(indices): # NOTE: NOT self.indices BUT THE PARAMETER 
            if (self.mode == "train") or (self.mode=="valid"): 
                x[i,] = img_to_array(load_img(
                    StanfordCarsDataloader.train_path_prefix +
                    StanfordCarsDataloader.trainDF.image[i],
                    target_size=self.data_shape, interpolation='bilinear'))

                if self.finegrained:
                    y1[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.trainDF.Class[i]]['broad_label'] - 1 
                    if (not self.broad_labels_only): y2[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.trainDF.Class[i]]['sub_label'] - 1 
                else:
                    y[i] = StanfordCarsDataloader.trainDF.Class[i] 


            elif self.mode == "test": 
                x[i,] = img_to_array(load_img(
                    StanfordCarsDataloader.test_path_prefix +
                    StanfordCarsDataloader.testDF.image[i],
                    target_size=self.data_shape, interpolation='bilinear'))
                
                if self.finegrained:
                    y1[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.testDF.Class[i]]['broad_label'] -1 
                    if (not self.broad_labels_only): y2[i] = StanfordCarsDataloader.labelDF.loc[StanfordCarsDataloader.testDF.Class[i]]['sub_label'] -1 
                else:
                    y[i] = StanfordCarsDataloader.testDF.Class[i] 
        
        if self.augment: 
            x = self.augment_func(x) 

        if self.finegrained:
            if (self.broad_labels_only): 
                return x, keras.utils.to_categorical(y1, num_classes = StanfordCarsDataloader.n_broad_labels) 
            
            if self.force_same_shape_sublabel: 
                if (self.broad_label_filter): 
                    return x, (keras.utils.to_categorical(y1, num_classes = StanfordCarsDataloader.n_broad_labels), 
                       keras.utils.to_categorical(y2, num_classes = StanfordCarsDataloader.ns_sub_labels[self.broad_label_filter]) ) 
                # else 
                return x, (keras.utils.to_categorical(y1, num_classes = StanfordCarsDataloader.n_broad_labels), 
                       keras.utils.to_categorical(y2, num_classes = max(StanfordCarsDataloader.ns_sub_labels)) ) 
            # else 
            return x, [(keras.utils.to_categorical(y1[idx], num_classes = StanfordCarsDataloader.n_broad_labels), 
                       keras.utils.to_categorical(y2[idx], num_classes = StanfordCarsDataloader.ns_sub_labels[y1[idx]]) 
                       ) for idx in range(len(y2))]
                       
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


