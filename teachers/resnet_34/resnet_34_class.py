import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tt
from torchvision import models
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models import StanfordCarsTeacherModel
import os

from .reqs import StanfordCarsModel 


class ResNet34Teacher(StanfordCarsTeacherModel):
    name = "resnet34" 
    pytorch = True 
    def __init__(self, num_classes = 196, teacher_state_dict_path = "teachers/resnet_34/stanfordcars-cnn.pth", use_cuda = torch.cuda.is_available()):  
        self.model = StanfordCarsModel(num_classes) 
        self.model.load_state_dict(torch.load(teacher_state_dict_path))

        if use_cuda: 
            self.model.to(torch.device('cuda'))


    test_transform = tt.Compose([
        tt.Resize((256,256)),
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(data, device):
        if isinstance(data,(list,tuple)):
            return [ResNet34Teacher.to_device(x,device) for x in data]
        return data.to(device, non_blocking=True)
    
    def predict(self, img): # return an int for the most likely label (0-1)
        self.model.eval()
        with torch.no_grad(): 
            device = ResNet34Teacher.get_default_device()

            # Convert to a batch of 1
            xb = ResNet34Teacher.to_device(img.unsqueeze(0), device)

            # Get predictions from model
            yb = self.model(xb)

            return yb[0].cpu() 

