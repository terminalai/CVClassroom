import sys
sys.path.append("../CVClassroom")
import numpy as np
import pandas as pd
import os
from teachers.CMAL_net_tresnet.CMAL_net_gating import CMALNetGatingModel 
import keras
from PIL import Image 
from data_generation import SoftlabelsDataloader as SLDL

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
        self.studentlst = [] 
        for i in range(1, self.n_broad_labels+1):
                self.studentlst.append(EfficientNetStudent(os.path.join(self.classdir, "expert_{}".format(i)), i, load_from_name=load_name))
    
    def save_to_name(self, save_name:str="main.keras"): 
        for i in range(self.n_broad_labels): 
            self.studentlst[i].save(save_name)
        

    def train(self, softlabels_file:str, train_aug_funcs:list=[add_gaussian_blur, lambda img: add_augmentation(img, default_target_img_shape[:2], 1)[0] ], # data noise is in aug_funcs 
                              valid_aug_funcs=[], valid_sampler=None, lr=1e-5, loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Accuracy(), keras.metrics.TopKCategoricalAccuracy(k=5)], 
                              num_epochs=1, ):
        i = 0 
        for mdl in self.studentlst:
            print("TRAINING {}".format(i))
            optimizer = keras.optimizers.AdamW(learning_rate=lr) 
            mdl.train_with_softlabels(softlabels_file, train_aug_funcs, valid_aug_funcs, valid_sampler, optimizer, loss, metrics, num_epochs=num_epochs) 
            i += 1 
    
    def setup_to_train_again(self): 
        for mdl in self.studentlst:
            mdl.model.trained_already = False 
            mdl.trained_already = False 

    def evaluate(self, softlabels_file:str, loss=keras.losses.BinaryCrossentropy()): 
        ress = [] 
        test_dl = SLDL("test", softlabels_file, mdl.expert_class, mdl.out_dim, False, 20, default_target_img_shape, [], valid_sampler=None, shuffle=True) 
        for mdl in self.studentlst: 

            metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.TopKCategoricalAccuracy(k=5)]

            for i in range(len(test_dl)): 
                x, y = test_dl[i]
                logits = self.model(x, training=False) 
                for metric in metrics: 
                    metric.update_state(y, logits) 
            metric_results = [] 
            for metric in metrics: 
                metric_result = metric.result() 
                metric_results.append(metric_result) 
                print("VAL METRIC", metric.name, "RESULT", metric_result)
                metric.reset_state() 

            ress.append(metric_results)
        return ress


    def predict_with_broad_label(self, img, broad_label): # input PIL image
        #assert img is not None, "No image detected for prediction"
        #print("IMG SHAPE PWBL", img.shape)
        return self.studentlst[broad_label].predict(img) 
    
    @classmethod 
    def get_gtg_model_broad_label(cls, img): 
        assert cls.gating_model is not None, "GATING MODEL NOT YET INITIALIZED FOR TrainingClassroom Class" 
        broad_label = np.argmax(cls.gating_model.broad_label_probs(keras.utils.load_img(img)))
        return broad_label
    
    def predict_with_gating_model(self, img):
        broad_label = self.get_gtg_model_broad_label(img)
        return self.studentlst[broad_label].predict(img) 



if __name__=="__main__": 
    # testing - only testing syntax for now 
    new_class = TrainingClassroom("./students/tcs/test_0", n_broad_labels=2) # to speed up the test 
    
    #new_class.train("./car_connection_dataset/cmalnet_softlabels_00.csv")

    # test isf it errors 
    import pandas as pd 
    labelDF = pd.read_csv("./car_connection_dataset/cmalnet_softlabels_00.csv")


    from keras.preprocessing.image import load_img, img_to_array
    from keras_cv.layers import RandAugment, RandomFlip, RandomRotation 
    import tensorflow as tf 

    # define augmentation function 
    default_augment_func = keras.Sequential([ RandAugment([0,255]), 
                                    RandomFlip(mode='horizontal'), 
                                    RandomRotation(factor=0.1), 
                                    ])
    
    img = default_augment_func(img_to_array(load_img(labelDF.loc[0, 'path'] , target_size=(224, 224, 3), interpolation='bilinear' ))) 

    img = tf.expand_dims(img, axis=0) 

    #print("IMG SHAPE", img.shape)

    print(new_class.predict_with_broad_label( img , 0)) 
#
    # test save/load 
    new_class.save_to_name('main.keras')

