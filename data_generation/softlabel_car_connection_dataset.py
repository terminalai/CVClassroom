# TODO: OPTIMIZE THIS by allowing batch predict 

import sys 
sys.path.append("../CVClassroom")

import os
dataset_path = "./car_connection_dataset/" 
_, dirs = os.walk(dataset_path)

import models 
from data_generation.stanford_cars_dataloader import StanfordCarsDataloader as SCDL 

from PIL import Image 
#import keras 
#from keras.preprocessing.image import load_img, img_to_array
from keras_cv.layers import RandAugment 
import numpy as np 
import params 


# define augmentation function 
rand_augment = RandAugment([0,255], **params.augment_params) 

def softlabel_with(model:models.StanfordCarsTeacherModel, #img_shape=(224, 224, 3), 
                   augment=True, augment_func=rand_augment, broad_conf_threshold=0.9, 
                   sub_conf_threshold=0.9, label_conf_threshold=0.81, labelling_mtd=None): 
    save_path = os.path.join(dataset_path, model.name+"_softlabels.csv") 
    with open(save_path, 'a+') as fout: 
        fout.write("path,broad_label,sub_label,label,label_probs\n") 
    
    if labelling_mtd: 
        SCDL.switch_labelling_method(labelling_mtd) 

    for name in dirs[2]:
        img_path = os.path.join(dataset_path, 'imgs', name) 
        #img = img_to_array(load_img(img_path, target_size=img_shape, interpolation='bilinear')) 
        #img = model.test_transform(img) 

        img = Image.open(img_path) 
        img = model.test_transform(img) 

        if augment: img = augment_func(img) 

        broad_label_probs = model.get_broad_label_probs(img) 
        if (broad_label_probs is None): 
            # implemented predict function instead 
            label_probs = model.predict(img) 
            #print(max(label_probs))
            likely_label = np.argmax(label_probs).item() 
            #print("LIKELY LABEL:", likely_label)
            #print(label_probs.shape)
            #print(label_probs)
            #print(label_probs[0])
            #print(label_probs[1])
            if label_probs[likely_label] < label_conf_threshold: continue 

            try: 
                likely_broad_label, likely_sub_label = SCDL.label_to_broad_sub_labels(likely_label+1) # since labels are 1-indexed 
            except Exception: 
                print("LIKELY LABEL:", likely_label) 

        else: 
            likely_broad_label = np.argmax(broad_label_probs).item() 
            if broad_label_probs[likely_broad_label] < broad_conf_threshold: continue 
            
            sub_label_probs = model.get_sub_label_probs(img, likely_broad_label) 
            likely_sub_label = np.argmax(sub_label_probs).item() 
            if sub_label_probs[likely_sub_label] < sub_conf_threshold: continue 

            likely_broad_label += 1 # 1-indexed labels 
            likely_sub_label += 1 
            likely_label = SCDL.broad_sub_labels_to_label(likely_broad_label, likely_sub_label)

        # save labels 
        with open(save_path, 'a+') as fout: 
            fout.write(str(img_path)) 
            fout.write(',') 
            fout.write(str(likely_broad_label)) 
            fout.write(',') 
            fout.write(str(likely_sub_label)) 
            fout.write(',') 
            fout.write(str(likely_label)) 
            fout.write(',') 
            fout.write('"'+str(label_probs.tolist())+'"')
            fout.write('\n') 



if __name__=='__main__': 
    #from teachers.CMAL_net_tresnet.CMAL_net_class import CMALNetTeacher 
    #softlabel_with(CMALNetTeacher(), augment=False) 

    from teachers.resnet_34.resnet_34_class import ResNet34Teacher 
    softlabel_with(ResNet34Teacher(), augment=False)

