import sys
sys.path.append("../CVClassroom")
import numpy as np
import pandas as pd
import os
from teachers.CMAL_net_tresnet.CMAL_net_gating import CMALNetGatingModel 
import keras
from PIL import Image 

from students.efficient_net_student import EfficientNetStudent, add_gaussian_blur, add_augmentation, default_target_img_shape 

def averagelabels(softlabel_files:list): # TODO: Create function that averages labels form multiple teachers 
    pass

class TrainingClassroom(): 

    gating_model = None 

    @classmethod 
    def initialize_cmanlet_gtg_model(cls): 
        cls.gating_model = CMALNetGatingModel()

    @classmethod 
    def add_gtg_model_softlabels(cls, in_file_path, out_file_path): 
        cls.gating_model is not None, "GATING MODEL NOT YET INITIALIZED FOR TrainingClassroom Class"
    
        labelDF = pd.read_csv(in_file_path)
        for i in range(len(labelDF)):
            labelDF.loc[i, "broad_label"] = 1+np.argmax(cls.gating_model.broad_label_probs( Image.open(labelDF.loc[i, "path"])) ) 
        labelDF.to_csv(out_file_path)

    
    def __init__(self, classroom_dir, n_broad_labels=16, load_names:str=None):
        self.classdir = classroom_dir
        self.studentlst = []
        self.n_broad_labels = n_broad_labels

        if not os.path.isdir(self.classdir): 
            os.makedirs(self.classdir) 


        if load_names: 
            self.load_from_name(load_names)
        else:         
            for i in range(1, n_broad_labels+1):
                self.studentlst.append(EfficientNetStudent(os.path.join(self.classdir, "expert_{}".format(i)), i))

    def load_from_name(self, load_name:str="main.keras"): 
        for i in range(1, self.n_broad_labels):
                self.studentlst.append(EfficientNetStudent(os.path.join(self.classdir, "expert_{}".format(i)), i, load_from_name=load_name))
    
    def save_to_name(self, save_name:str="main.keras"): 
        for i in range(self.n_broad_labels): 
            self.studentlst[i].save(save_name)
        

    def train(self, softlabels_file:str, train_aug_funcs:list=[add_gaussian_blur, lambda img: add_augmentation(img, default_target_img_shape[:2], 1)[0] ], # data noise is in aug_funcs 
                              valid_aug_funcs=[], valid_sampler=None, optimizer='AdamW', loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Accuracy(), keras.metrics.TopKCategoricalAccuracy(k=5)]):
        for mdl in self.studentlst:
            mdl.train_with_softlabels(softlabels_file, train_aug_funcs, valid_aug_funcs, valid_sampler, optimizer, loss, metrics) 


    def predict_with_broad_label(self, img, broad_label): # input PIL image
        #assert img is not None, "No image detected for prediction"
        print("IMG SHAPE PWBL", img.shape)
        return self.studentlst[broad_label].predict(img) 
    
    @classmethod 
    def get_gtg_model_broad_label(cls, img): 
        assert cls.gating_model is not None, "GATING MODEL NOT YET INITIALIZED FOR TrainingClassroom Class" 
        broad_label = np.argmax(cls.gating_model.broad_label_probs(keras.utils.load_img(img)))
        return broad_label
    



if __name__=="__main__": 
    # testing - only testing syntax for now 
    new_class = TrainingClassroom("./students/tcs/test_0", n_broad_labels=2) # to speed up the test 
    
    #new_class.train("./car_connection_dataset/cmalnet_softlabels_00.csv")

    # test isf it errors 
    import pandas as pd 
    labelDF = pd.read_csv("./car_connection_dataset/cmalnet_softlabels_00.csv")


    from keras.preprocessing.image import load_img, img_to_array
    from keras_cv.layers import RandAugment, RandomFlip, RandomRotation 

    # define augmentation function 
    default_augment_func = keras.Sequential([ RandAugment([0,255]), 
                                    RandomFlip(mode='horizontal'), 
                                    RandomRotation(factor=0.1), 
                                    ])
    
    img = default_augment_func(img_to_array(load_img(labelDF.loc[0, 'path'] , target_size=(224,224,3), interpolation='bilinear' ))) 

    print("IMG SHAPE", img.shape)

    print(new_class.predict_with_broad_label( img , 0)) 

    # test save/load 
    new_class.save()

