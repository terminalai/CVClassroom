# IMPORTS
import sys 
sys.path.append("../CVClassroom")

import os 

from models import StanfordCarsStudentModel
from models import EfficientNetModel
from data_generation import SoftlabelsDataloader as SLDL
from data_noise import add_gaussian_blur, add_augmentation

import keras 



# VARIABLES
default_target_img_shape = (224, 224, 3) 
n_classes_per_broad_label = [14, 10, 14, 13, 14, 14, 15, 12, 15, 12, 11, 11, 11, 11, 9, 10]

# FUNCTIONS
def fetch_valid(indices, train_over_total_frac): return indices[:int(len(indices)*train_over_total_frac)] # Basically a validation sampler (if i understood what valid sampler is correctly)

# CLASS
class EfficientNetStudent(StanfordCarsStudentModel, EfficientNetModel):
    
    def __init__(self, save_dir, expert_class, # expert_class is 1-Indexed
                 noise_type='ltnl', img_shape = default_target_img_shape, load_from_name=None):
        # initialise abstract super classes
        StanfordCarsStudentModel.__init__(self, expert_class, img_shape)
        self.out_dim = n_classes_per_broad_label[expert_class-1] 
        if expert_class is None: self.out_dim = 196
        EfficientNetModel.__init__(self, save_dir, self.out_dim, img_shape=img_shape, noise_type=noise_type)
        if load_from_name: 
            self.model.load_weights(os.path.join(save_dir, load_from_name))

    def train_with_softlabels(self, softlabels_file:str, train_aug_funcs:list=[add_gaussian_blur, lambda img: add_augmentation(img, default_target_img_shape[:2], 1)[0] ], # data noise is in aug_funcs 
                              valid_aug_funcs=[], valid_sampler=None, optimizer=keras.optimizers.AdamW(learning_rate=1e-5), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Accuracy(), keras.metrics.TopKCategoricalAccuracy(k=5)], 
                              num_epochs=12, valid_freq=3,): 
        train_DL = SLDL("train", softlabels_file, self.expert_class, self.out_dim, False, 200, default_target_img_shape, train_aug_funcs, valid_sampler=valid_sampler, shuffle=True) # mode, labelfile, expert_class, use_gating_mdl, batchsize, datashape, modification_functions, no_train_indices_in_valid?, validation sampler, shuffle,**kwargs
        valid_DL = SLDL("valid", softlabels_file, self.expert_class, self.out_dim,False, 200, default_target_img_shape, valid_aug_funcs, valid_sampler=valid_sampler, shuffle=True)

        EfficientNetModel.train(self, train_DL, valid_DL, optimizer=optimizer, loss=loss, metrics=metrics, num_epochs=num_epochs, valid_freq=valid_freq, )

        # clear 
        del train_DL 
        del valid_DL 

    # Overloaded function, train dataloaders
    def train_with_dataloaders(self, train_DL, valid_DL, optimizer='AdamW', loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Accuracy(), keras.metrics.TopKCategoricalAccuracy(k=5)], 
                              num_epochs=12, valid_freq=3, callbacks = None, compile_kwargs={},):
        EfficientNetModel.train(self, train_DL, valid_DL, optimizer=optimizer, loss=loss, metrics=metrics, num_epochs=num_epochs, valid_freq=valid_freq, callbacks=callbacks, compile_kwargs=compile_kwargs)

    def predict(self, img):
        return EfficientNetModel.predict(self, img)


# test if this works or not 
if __name__=="__main__": 
    ens = EfficientNetStudent('.', 1, )
    ens.train_with_softlabels('./car_connection_dataset/cmalnet_softlabels_00.csv')
