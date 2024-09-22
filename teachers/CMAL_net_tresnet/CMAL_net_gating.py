from .CMAL_net_class import CMALNetTeacher 
from models.modelss import StanfordCarsGatingModel 
from data_generation.stanford_cars_dataloader import StanfordCarsDataloader as SCDL 
from data_generation.torch_stanford_cars_dataloader import get_dataloaders 

import numpy as np 


class CMALNetGatingModel(CMALNetTeacher, StanfordCarsGatingModel): 
    # Base class for teacher models for stanford cars dataset, to be inherited from. 
    # for both pytorch and keras 

    name = "cmalnet_gatingmodel" 


    # GET BROAD LABEL

    def broad_label_probs(self, img): # returns an probabilities for each broad label 
        label_probs = self.predict(img) 
        return [label_probs[np.array(SCDL.labels_for_broad_label(bl))-1].sum().item() for bl in range(SCDL.n_broad_labels)] 



if __name__ == "__main__": 
    teacher1 = CMALNetGatingModel() 
    teacher2 = CMALNetGatingModel() 
    traindl, testdl = get_dataloaders(batch_size=1, test_transforms = CMALNetGatingModel.test_transform) 
    imgs, ans = next(iter(testdl)) 
    #print(imgs) 
    #print(ans) 
    #print(teacher1.net)
    #teacher1.net.train() 
    print(teacher1.broad_label_probs(imgs[0:1].clone().detach())) 
    res2 = teacher2.broad_label_probs(imgs[0:1].clone().detach()) 
    print(res2) 
