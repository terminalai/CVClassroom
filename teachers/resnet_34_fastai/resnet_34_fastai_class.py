# uncommenting the commented libraries will require new requirements in requirements.txt 

#import seaborn as sns
from fastai.vision.all import *
from fastai.metrics import error_rate
#from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from glob import iglob
#import cv2
#from pathlib import Path
from models import StanfordCarsTeacherModel

class Resnet34FastaiModel(StanfordCarsTeacherModel): 
    # Base class for teacher models for stanford cars dataset, to be inherited from. 
    # for both pytorch and keras 

    name = "resnet34fastai" 
    pytorch = False 
       
    
    def __init__(self, teacher_path = "teachers/resnet_34_fastai/ResNet34_phase6.h5.pth"): 
        learn = learn.load(teacher_path)

    def test_transform(img): 
        return img 

    # predict functions return in the form: (name of vehicle, label index (as a tensor), list of probabilities)
    # predict functions require path of the img, idk how to make it such that it accepts an 2d array of values (may be a good or bad thing)
    def predict(self, img_path_dict): # for many imgs at once, I could optimize it (find a batch predict fn)
        preds = []
        for img_path in img_path_dict:
            pred = self.learn.predict(img_path)
            preds.append(pred)
        return preds
    
    def predict(self, img_path): 
        pred = self.learn.predict(img_path)
        return pred
