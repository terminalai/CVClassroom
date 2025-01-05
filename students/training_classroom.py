import sys
sys.path.append("../CVClassroom")
import numpy as np
import pandas as pd
import os
from tempfile import NamedTemporaryFile
from teachers.CMAL_net_tresnet.CMAL_net_gating import CMALNetGatingModel as GM
import keras

from students.efficient_net_student import EfficientNetStudent

def averagelabels(softlabel_files:list): # TODO: Create function that averages labels form multiple teachers 
    pass

class TrainingClassroom(): 
    
    def __init__(self, classroom_dir, softlabels_file, n_broad_labels=16, use_gating_model=False):
        self.classdir = classroom_dir
        self.labeldir = softlabels_file
        self.use_gtg = use_gating_model
        self.studentlst = []
        
        for i in range(1, n_broad_labels+1):
            self.studentlst.append(EfficientNetStudent(sys.append(self.classdir, "-", str(i)), i))
        
        if self.use_gtg:
            labelDF = pd.read_csv(self.labeldir)
            for i in range(len(labelDF)):
                labelDF.loc[i, "broad_label"] = 1+np.argmax(GM.get_broad_labels(keras.utils.load_img(os.path.join(self.data_folder_prefix , self.labelDF.loc[i, "path"]))))
            self.tl = NamedTemporaryFile(suffix=".csv", delete=False)
            labelDF.to_csv(self.tl.name)
            self.labeldir = self.tl.name


    def train(self):
        for mdl in self.studentlst:
            mdl.train_with_softlabels(self.labeldir, self.use_gtg)

        if self.use_gtg:
            self.tl.close()
            os.unlink(self.tl.name)


    def predict(self, img): # input PIL image
        assert img is not None, "No image detected for prediction"
        expert_class_index = np.argmax(GM.get_broad_labels(keras.utils.load_img(img)))
        model = self.studentlst[expert_class_index]
        return model.predict(img)



# MAIN
new_class = TrainingClassroom("", "", use_gating_model=True) # please add directory and file paths accordingly
new_class.train()
print(new_class.predict(None)) # add image
