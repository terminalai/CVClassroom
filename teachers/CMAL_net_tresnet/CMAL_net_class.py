from models import StanfordCarsTeacherModel 

import torch 
from torch.autograd import Variable 
from torchvision import transforms 

class CMALNetTeacher(StanfordCarsTeacherModel): 

    name = "cmalnet" 
    pytorch = True 



    def __init__(self): 
        self.net = 1 # TODO 
    
    test_trainsform = transforms.ToTensor() 


    @classmethod  
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
    
    @classmethod 
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


    def predict(self, inputs, use_cuda = torch.cuda.is_available()): 
        self.net.eval()
        
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

        return outputs_com # OUTPUT FORMAT: TODO figure out output format ...? 




