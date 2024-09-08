from fastai.vision.all import *
from models import StanfordCarsTeacherModel
#from keras.preprocessing.image import save_img
#from PIL import Image
from tempfile import NamedTemporaryFile

class Resnet34FastaiModel(StanfordCarsTeacherModel): 
    # Base class for teacher models for stanford cars dataset, to be inherited from. 
    # for both pytorch and keras 

    name = "resnet34fastai" 
    pytorch = False 
       
    
    def __init__(self, teacher_path = "teachers/resnet_34_fastai/ResNet34_phase6.h5.pth"): 
        
        self.model = load_learner(teacher_path)

    def test_transform(img): # input: pillow image 
        return img 

    

    # TODO: maybe find a batch predict fn (though will also have to change the softlabeller code) 

    
    def predict(self, image, return_all:bool=False): # takes in image as pillow image 
        # return_all means in the form: (name of vehicle, label index (as a tensor), list of probabilities)
        # otherwise just all probabilities 

        with NamedTemporaryFile(suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            pred = self.model.predict(temp_file.name)
        
        if return_all: 
            return pred 
        
        return pred[2] 

    def predict_from_filename(self, filename): 
        return self.learn.predict(filename)[2] 
            

