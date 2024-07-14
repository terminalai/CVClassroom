import sys 
sys.path.insert(1, sys.path[0]+'/../../')

import os 
sys.path.insert(1, os.curdir)

from data_generation.torch_stanford_cars_dataloader import get_dataloaders   

from models import StanfordCarsTeacherModel 

from teachers.CMAL_net_tresnet.src.models.tresnet_v2.tresnet_v2 import TResnetL_V2

import torch 
import torch.nn as nn 
from torch.autograd import Variable 
from torchvision import transforms 

from .basic_conv import BasicConv 


class Features(nn.Module):
    def __init__(self, net_layers_FeatureHead):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers_FeatureHead[0])
        self.net_layer_1 = nn.Sequential(*net_layers_FeatureHead[1])
        self.net_layer_2 = nn.Sequential(*net_layers_FeatureHead[2])
        self.net_layer_3 = nn.Sequential(*net_layers_FeatureHead[3])
        self.net_layer_4 = nn.Sequential(*net_layers_FeatureHead[4])
        self.net_layer_5 = nn.Sequential(*net_layers_FeatureHead[5])


    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x1 = self.net_layer_3(x)
        x2 = self.net_layer_4(x1)
        x3 = self.net_layer_5(x2)

        return x1, x2, x3


class Network_Wrapper(nn.Module):
    def __init__(self, net_layers,num_classes):
        super().__init__()
        self.Features = Features(net_layers)

        self.max_pool1 = nn.MaxPool2d(kernel_size=46, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=23, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=12, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )



    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        #print('x1.shape', x1.shape)
        x1_ = self.conv_block1(x1)
        map1 = x1_.clone().detach()
        #print('x1_.shape', x1_.shape)
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)

        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        map2 = x2_.clone().detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        map3 = x3_.clone().detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3

import os 
import requests 

class CMALNetTeacher(StanfordCarsTeacherModel): 

    name = "cmalnet" 
    pytorch = True 



    def __init__(self, model_params = {'num_classes': 196}, teacher_state_dict_path = "teachers/CMAL_net_tresnet/model_state_dict.pth", use_cuda = torch.cuda.is_available()): 
        '''model = TResnetL_V2(model_params)
        pretrained_weights = torch.load(weights_path)
        model.load_state_dict(pretrained_weights.load_state_dict())

        net_layers = list(model.children())
        net_layers = net_layers[0]
        net_layers = list(net_layers.children())
        
        self.net = Network_Wrapper(net_layers, 196)
        if use_cuda: 
            self.net.to(torch.device('cuda'))'''
        model = TResnetL_V2(model_params)
        weights_url = \
            'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/tresnet/stanford_cars_tresnet-l-v2_96_27.pth'
        weights_path = "tresnet-l-v2.pth"

        if not os.path.exists(weights_path):
            print('downloading weights...')
            r = requests.get(weights_url)
            with open(weights_path, "wb") as code:
                code.write(r.content)
        pretrained_weights = torch.load(weights_path)
        model.load_state_dict(pretrained_weights['model'])
            
        net_layers = list(model.children())
        net_layers = net_layers[0]
        net_layers = list(net_layers.children())
        
        self.net = Network_Wrapper(net_layers, 196)

        self.net.load_state_dict(torch.load(teacher_state_dict_path)) 
        if use_cuda: 
            self.net.to(torch.device('cuda'))
    

        self.model_params = model_params 
        self.weights_path = weights_path 

    
    test_transform = transforms.Compose([
        transforms.Resize((421, 421)),
        transforms.RandomCrop(368, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    def map_generate(attention_map, pred, p1, p2):
        batches, feaC, feaH, feaW = attention_map.size()

        out_map=torch.zeros_like(attention_map.mean(1))

        for batch_index in range(batches):
            map_tpm = attention_map[batch_index]
            map_tpm = map_tpm.reshape(feaC, feaH*feaW)
            map_tpm = map_tpm.permute([1, 0])
            p1_tmp = p1.permute([1, 0])
            map_tpm = torch.mm(map_tpm, p1_tmp)
            map_tpm = map_tpm.permute([1, 0])
            map_tpm = map_tpm.reshape(map_tpm.size(0), feaH, feaW)

            pred_tmp = pred[batch_index]
            pred_ind = pred_tmp.argmax()
            p2_tmp = p2[pred_ind].unsqueeze(1)

            map_tpm = map_tpm.reshape(map_tpm.size(0), feaH * feaW)
            map_tpm = map_tpm.permute([1, 0])
            map_tpm = torch.mm(map_tpm, p2_tmp)
            out_map[batch_index] = map_tpm.reshape(feaH, feaW)

        return out_map
    
    def highlight_im(images, attention_map, attention_map2, attention_map3, theta=0.5, padding_ratio=0.1):
        images = images.clone()
        attention_map = attention_map.clone().detach()
        attention_map2 = attention_map2.clone().detach()
        attention_map3 = attention_map3.clone().detach()

        batches, _, imgH, imgW = images.size()

        for batch_index in range(batches):
            image_tmp = images[batch_index]
            map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
            map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
            map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


            map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
            map_tpm2 = torch.nn.functional.upsample_bilinear(map_tpm2, size=(imgH, imgW)).squeeze()
            map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

            map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
            map_tpm3 = torch.nn.functional.upsample_bilinear(map_tpm3, size=(imgH, imgW)).squeeze()
            map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

            map_tpm = (map_tpm + map_tpm2 + map_tpm3)
            map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
            map_tpm = map_tpm >= theta

            nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
            image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

            images[batch_index] = image_tmp

        return images

    softmax = nn.Softmax() 

    def predict(self, inputs, use_cuda = torch.cuda.is_available()): 
        self.net.eval()

        with torch.no_grad(): 

            if len(inputs.shape)==3: 
                inputs = inputs[None,:,:,:] 
            
            if use_cuda:
                device = torch.device("cuda")
                inputs = inputs.to(device) 
            inputs = Variable(inputs, volatile=True) 
            output_1, output_2, output_3, output_concat, map1, map2, map3 = self.net(inputs)

            p1 = self.net.state_dict()['classifier3.1.weight']
            p2 = self.net.state_dict()['classifier3.4.weight']
            att_map_3 = CMALNetTeacher.map_generate(map3, output_3, p1, p2)

            p1 = self.net.state_dict()['classifier2.1.weight']
            p2 = self.net.state_dict()['classifier2.4.weight']
            att_map_2 = CMALNetTeacher.map_generate(map2, output_2, p1, p2)

            p1 = self.net.state_dict()['classifier1.1.weight']
            p2 = self.net.state_dict()['classifier1.4.weight']
            att_map_1 = CMALNetTeacher.map_generate(map1, output_1, p1, p2)

            inputs_ATT = CMALNetTeacher.highlight_im(inputs, att_map_1, att_map_2, att_map_3)
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = self.net(inputs_ATT)

            outputs_com2 = output_1 + output_2 + output_3 + output_concat
            outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

        return CMALNetTeacher.softmax(outputs_com.cpu()[0]) 



if __name__ == "__main__": 
    teacher1 = CMALNetTeacher() 
    teacher2 = CMALNetTeacher() 
    traindl, testdl = get_dataloaders(batch_size=1, test_transforms = CMALNetTeacher.test_transform) 
    imgs, ans = next(iter(testdl)) 
    #print(imgs) 
    #print(ans) 
    #print(teacher1.net)
    #teacher1.net.train() 
    print(teacher1.predict(imgs[0:1].clone().detach())) 
    res2 = teacher2.predict(imgs[0:1].clone().detach()) 
    print(res2) 

