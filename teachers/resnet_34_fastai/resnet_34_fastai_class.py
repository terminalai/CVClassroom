import sys 
sys.path.append("../CVClassroom")

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
       
    
    def __init__(self, teacher_path = "teachers/resnet_34_fastai/resnet_34_fastai_model.pth"): 
        
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
        return self.model.predict(filename)[2] 
            

if __name__ == "__main__": 
    from data_generation.torch_stanford_cars_dataloader import get_dataloaders  

    teacher1 = Resnet34FastaiModel() 
    teacher2 = Resnet34FastaiModel() 
    traindl, testdl = get_dataloaders(batch_size=1, test_transforms = Resnet34FastaiModel.test_transform) 
    imgs, ans = next(iter(testdl)) 
    #print(imgs) 
    #print(ans) 
    #print(teacher1.net)
    #teacher1.net.train() 
    print(teacher1.predict(imgs[0:1].clone().detach())) 
    res2 = teacher2.predict(imgs[0:1].clone().detach()) 
    print(res2) 

