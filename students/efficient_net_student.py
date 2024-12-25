# IMPORTS
import sys 
sys.path.append("../CVClassroom")

from models import StanfordCarsStudentModel
from models import EfficientNetModel
import pandas as pd
from data_generation import SoftlabelsDataloader as SLDL
from data_noise import add_gaussian_blur, add_augmentation

# VARIABLES
default_target_img_shape = (224, 224, 3) 
n_classes_per_broad_label = [14, 10, 14, 13, 14, 14, 15, 12, 15, 12, 11, 11, 11, 11, 9, 10]

class EfficientNetStudent(StanfordCarsStudentModel, EfficientNetModel):
    
    def __init__(self, save_dir, data_dir, label_file, expert_class, # expert_class is 1-Indexed
                 mdl_noise = True, img_shape = default_target_img_shape):
        # TODO: implement option to include data and model noise or not
        # initialise abstract super classes
        StanfordCarsStudentModel.__init__(self, expert_class, img_shape)
        EfficientNetModel.__init__(self, save_dir, n_classes_per_broad_label[expert_class-1], img_shape=img_shape)

        # create train and valid dataframes 
        self.train_DL = SLDL.__init__("train", data_dir, label_file, expert_class, True, 20, default_target_img_shape, True, [add_gaussian_blur, add_augmentation(img_size=default_target_img_shape[0])], split_train_valid(train_over_total_frac=0.8), shuffle=True) # mode, datafile, labelfile, expert_class, no_train_indices_in_valid?, batchsize, datashape, modify_data?, modification_functions, validation sampler, shuffle,**kwargs
        self.valid_DL = SLDL.__init__("valid", data_dir, label_file, expert_class, True, 20, default_target_img_shape, True, [add_gaussian_blur, add_augmentation(img_size=default_target_img_shape[0])], split_train_valid(train_over_total_frac=0.8), shuffle=True)
    
    def train(self, optimizer='AdamW', loss=..., metrics=..., num_epochs=12, valid_freq=3, callbacks = None, compile_kwargs=..., **fit_kwargs):
        EfficientNetModel.train(self, self.train_DL, self.valid_DL, optimizer=optimizer, loss=loss, metrics=metrics, num_epochs=num_epochs, valid_freq=valid_freq, callbacks=callbacks, compile_kwargs=compile_kwargs)

    def predict(self, img):
        return EfficientNetModel.predict(self, img)

# FUNCTIONS

def split_train_valid(indices, train_over_total_frac): # TODO: create a function that splits a .csv or list of .csv files into train and valid dataloaders

    idx = int(len(indices)*train_over_total_frac)
    valid = indices[idx:]
    return valid
    