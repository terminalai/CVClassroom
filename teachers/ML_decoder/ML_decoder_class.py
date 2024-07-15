import sys 
sys.path.insert(1, sys.path[0]+'/../../')

import torch 
from torchvision import transforms 

import numpy as np 

from src_files.helper_functions.bn_fusion import fuse_bn_recursively 
from src_files.models.tresnet.tresnet import InplacABN_to_ABN, TResnetL 
from models import StanfordCarsTeacherModel 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 


class MLDecoderModel(StanfordCarsTeacherModel): 
    # Base class for teacher models for stanford cars dataset, to be inherited from. 
    # for both pytorch and keras 

    name = "mldecoder" 

    pytorch = True 

    test_transform = transforms.Compose([]) #transforms.Compose([transforms.ToTensor(), transforms.Resize((384, 384))]) # TODO 
    # note that this takes in a pillow image, so it must start with transforms.ToTensor() 
    # this will be used in the torch Dataset or DataLoader 
    
    
    def __init__(self): 
        self.model = TResnetL({'num_classes':196}) 

        self.state = torch.load('tresnet_l_stanford_card_96.41.pth', map_location='cpu') 
        self.model.load_state_dict(self.state['model'], strict=True)
        # eliminate BN for faster inference 
        model = model.cpu() 
        model = InplacABN_to_ABN(model) 
        model = fuse_bn_recursively(model) 
        model = model.cuda().half().eval() 
    


    def predict(self, img): # returns an probabilities for each label (0-195) 
        im_resize = img.resize((384, 384))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
        tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
        output = torch.squeeze(torch.sigmoid(self.model(tensor_batch)))
        np_output = output.cpu().detach().numpy()

        return np_output 


        ## Top-k predictions
        # detected_classes = classes_list[np_output > args.th]
        idx_sort = np.argsort(-np_output)
        detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
        scores = np_output[idx_sort][: args.top_k]
        idx_th = scores > args.th
        detected_classes = detected_classes[idx_th]
        print('done\n')



if __name__ == "__main__": 
    from data_generation.torch_stanford_cars_dataloader import get_dataloaders   
    teacher1 = MLDecoderModel() 
    teacher2 = MLDecoderModel() 
    traindl, testdl = get_dataloaders(batch_size=1, test_transforms = MLDecoderModel.test_transform) 
    imgs, ans = next(iter(testdl)) 
    #print(imgs) 
    #print(ans) 
    #print(teacher1.net)
    #teacher1.net.train() 
    print(teacher1.predict(imgs[0:1].clone().detach())) 
    res2 = teacher2.predict(imgs[0:1].clone().detach()) 
    print(res2) 


