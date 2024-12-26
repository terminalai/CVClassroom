# IMPORTS
import os 
import sys
sys.path.append("../CVClassroom")
from teachers.CMAL_net_tresnet.CMAL_net_gating import CMALNetGatingModel as GM
import pandas as pd
import numpy as np
import keras.utils

# VARIABLES
seed = 12; np.random.seed(seed)

# FUNCTIONS
def default_vsampler(indices): return indices[int(len(indices)*0.8):]

# DATALOADER
class SoftlabelsDataloader(keras.utils.PyDataset):
    

    def __init__(self, mode, labels_file:str, expert_class=None, out_dim=10, use_gating_mdl=False, batch_size=20, data_shape=(224, 224, 3), mod_functions:list=[],  
                 valid_sampler=None, shuffle=True, data_folder_prefix = '', return_main_label=False, one_hot=True, **kwargs): 
        # if expert_class is None, then it loads data for all classes and not only for that specific broad label 
        # in that case, the y of the data will have both broad and sub labels. 

        # y will be (batch_size, 2/3) or just (batch_size) 
        # order will be (broad_label, sub_label, main_label) 
        # if no broad label (as expert is set) or no main label (if return_main_label=False), then it will be less than those 3 

        if valid_sampler is None: 
            valid_sampler = default_vsampler 

        # initialise object attributes
        self.bs = batch_size
        self.dts = data_shape
        self.modfuncs = mod_functions # input/output format: PIL Image
        self.shuffle = shuffle
        self.data_folder_prefix = data_folder_prefix 
        self.return_main_label = return_main_label 
        self.one_hot = one_hot # only of expert_class is set and return_main_label=False 
        self.out_dim=out_dim # only used if one_hot=True
        self.expert_class = expert_class

        assert (not self.one_hot) or ((expert_class is not None) and (not return_main_label)) , "One-hot encoding is only available when the expert class is set and the main label is not to be returned! (SLDL)"
        
        # initialise PyDataset
        super().__init__(**kwargs)

        # fetch labels
        self.labelDF = pd.read_csv(labels_file)

        # check validity of mode
        assert ((mode == "train") or (mode == 'valid') or (mode == 'test')), 'That data loading mode is not available.'

        # filter data based on mode
        self.indices = self.labelDF.index.to_list() 
        
        # shuffle data
        if self.shuffle:
            np.random.shuffle(self.indices)

        if mode == "train": 
            temp = valid_sampler(self.indices)
            self.indices = list(filter(lambda item: item not in temp, self.indices))
            del temp
        elif mode == "valid":
            self.indices = valid_sampler(self.indices)
        
        if expert_class is not None: 
            # filter data based on expert class and convert to numpy array
            if use_gating_mdl: self.indices = np.array(list(filter(lambda x: expert_class == np.argmax(GM.get_broad_labels(keras.utils.load_img( os.path.join(self.data_folder_prefix , self.labelDF.loc[x, "path"])))), self.indices)))
            else: self.indices = np.array(list(filter(lambda x: expert_class == self.labelDF.loc[x, "broad_label"], self.indices)))
 

    def on_epoch_end(self): 
        if self.shuffle:
            np.random.shuffle(self.indices)


    def create_batch(self, indices): # generates data given some indices, not the entire thing 
        # generate numpy arrays for the images and labels
        x = np.empty((self.bs, *self.dts)) # images
        
        n = 1 
        if self.one_hot: 
            y = np.zeros( (self.bs, self.out_dim), dtype=int) # labels 
        else: 
            if self.expert_class is None: n += 1 
            if self.return_main_label: n += 1 
            if n>1: 
                y = np.empty( (self.bs, n) , dtype=int)
            else: 
                y = np.empty(self.bs, dtype=int) # labels 

        # convert image to array and fill numpy arrays with data and labels
        num_finished = 0 
        for i in indices:
            # get image
            img = keras.utils.load_img( os.path.join(self.data_folder_prefix , self.labelDF.loc[i, "path"]) , target_size=self.dts, interpolation='bilinear')
            
            # modify the image
            if len(self.modfuncs):
                for func in self.modfuncs:
                    img = func(img)
            
            # add image to created batch
            x[num_finished,] = keras.utils.img_to_array(img)

            if n==1: 
                if self.one_hot: 
                    y[num_finished, self.labelDF.loc[i, "sub_label"]-1] = 1 # since sub label is 1-indexed, so -1 
                else: 
                    y[num_finished] = self.labelDF.loc[i, "sub_label"] 
            else: 
                alr = 0 
                if self.expert_class is None: 
                    y[num_finished, alr] = self.labelDF.loc[i, "broad_label"]
                    alr += 1 
                
                y[num_finished, alr] = self.labelDF.loc[i, "sub_label"]
                alr += 1 

                if self.return_main_label: 
                    y[num_finished, alr] = self.labelDF.loc[i, "label"]                

            num_finished += 1 
            
        # return created batch
        return x, y


    def __len__(self): 
        return int(len(self.indices) / self.bs)


    def __getitem__(self, index): # get the batch of data 
        indices = self.indices[index * self.bs : (index+1) * self.bs] 

        return self.create_batch(indices) 



# test that it works 
if __name__=='__main__': 

    tdl = SoftlabelsDataloader('train', './car_connection_dataset/cmalnet_softlabels_00.csv', batch_size=16) 
    vdl = SoftlabelsDataloader('valid', './car_connection_dataset/cmalnet_softlabels_00.csv', batch_size=16) 

    a, b = tdl[0]
    print(a.shape) 
    print(b.shape)
    print("TDL:", len(tdl)) 
    print("VDL:", len(vdl)) 

    print("TDL2:")
    tdl2 = SoftlabelsDataloader('train', './car_connection_dataset/cmalnet_softlabels_00.csv', batch_size=8, expert_class=2, return_main_label=True) 
    for i in range(3): 
        _, t = tdl2[i] 
        print(t) 

    print(tdl.labelDF[tdl.labelDF['broad_label']==2][['sub_label', 'label']]) 

