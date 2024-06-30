import sys 
sys.path.append("../CVClassroom")

import os
dataset_path = "../car_connection_dataset/" 
_, dirs = os.walk(dataset_path)

import models 

#import keras 
from keras.preprocessing.image import load_img, img_to_array
from keras_cv.layers import RandAugment 
import numpy as np 
import params 


# define augmentation function 
rand_augment = RandAugment([0,255], **params.augment_params) 

def softlabel_with(model:models.StanfordCarsModel, img_shape=(224, 224, 3), 
                   augment=True, augment_func=rand_augment, broad_conf_threshold=0.9, 
                   sub_conf_threshold=0.9, ): 
    save_path = os.path.join(dataset_path, model.name+"_softlabels.csv") 
    with open(save_path, 'a+') as fout: 
        fout.write("path,broad_label,sub_label\n") 

    for name in dirs[2]:
        img_path = os.path.join(dataset_path, 'imgs', name) 
        img = img_to_array(load_img(img_path, target_size=img_shape, interpolation='bilinear')) 
        if augment: img = augment_func(img) 

        broad_label_probs = model.get_broad_label_probs(img) 
        likely_broad_label = np.argmax(broad_label_probs) 
        if broad_label_probs[likely_broad_label] < broad_conf_threshold: continue 
        
        sub_label_probs = model.get_sub_label_probs(img, likely_broad_label) 
        likely_sub_label = np.argmax(sub_label_probs) 
        if sub_label_probs[likely_sub_label] < sub_conf_threshold: continue 

        # save labels 
        with open(save_path, 'a+') as fout: 
            fout.write(str(img_path)) 
            fout.write(',') 
            fout.write(str(likely_broad_label)) 
            fout.write(',') 
            fout.write(str(likely_sub_label)) 
            fout.write('\n') 



# TODO 

# have some function for loading car connection dataset to be softlabelled; smtg must create [TEACHER_NAME]_softlabels.csv file as described below. Maybe that function shld be somewhere 
