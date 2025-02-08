import os
os.environ['KERAS_BACKEND'] = "tensorflow" 

from training_classroom import TrainingClassroom 
import keras 
from data_generation import softlabels_dataloader as SLDL 

import os 
for expert_num in range(1, 17): 
    try: 
        os.makedirs("./students/tcs/test_0/expert_{}/".format(expert_num))
    except: 
        pass 



# testing - only testing syntax for now 
tc = TrainingClassroom("./students/tcs/test_0", n_broad_labels=16) # to speed up the test 

#new_class.train("./car_connection_dataset/cmalnet_softlabels_00.csv")

# test isf it errors 
import pandas as pd 
labelDF = pd.read_csv("./car_connection_dataset/cmalnet_softlabels_00.csv")


tc.load_from_name("epoch_6.keras")

epoch = 6
#tc.save_to_name("epoch_{}.keras".format(epoch))
while True: 
    tc.train('./car_connection_dataset/cmalnet_softlabels_00.csv', lr=1e-5, num_epochs=6) 
    tc.setup_to_train_again() 

    # test 
    try: 
        tc.evaluate('./car_connection_dataset/cmalnet_softlabels_00.csv')
    except Exception as e: 
        print("EVALUATION FAILED UGH")
        print(e) 

    epoch += 6
    tc.save_to_name("epoch_{}.keras".format(epoch))
    

