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

class ResNet34Teacher(StanfordCarsTeacherModel):
    name = "resnet34" 
    pytorch = True 
    def __init__(self, num_classes = 196, teacher_state_dict_path = "teachers/resnet_34/stanfordcars-cnn.pth", use_cuda = torch.cuda.is_available()): 
        super.__init__()        
        model = models.resnet34(pretrained=True)
        self.net = model
        self.net.fc = nn.Linear(self.net.fc.in_features,num_classes)
        self.net.load_state_dict(torch.load(teacher_state_dict_path))
        weights_path = 'weights/stanfordcars-cnn.pth'

        # if not os.path.exists(weights_path):
        #     print('downloading weights...')
        #     r = requests.get(weights_url)
        #     with open(weights_path, "wb") as code:
        #         code.write(r.content)

        pretrained_weights = torch.load(weights_path)
        model.load_state_dict(pretrained_weights['model'])

        if use_cuda: 
            self.net.to(torch.device('cuda'))

        self.weights_path = weights_path 

    test_tfms = tt.Compose([
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
        self.net.eval()
        device = ResNet34Teacher.get_default_device()

        # Convert to a batch of 1
        xb = ResNet34Teacher.to_device(img.unsqueeze(0), device)

        # Get predictions from model
        yb = self.net(xb)

        return yb 

        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)

        # Return class integer label
        return preds[0].item()