import keras 
import math 
import copy 

class SaveEveryEpochCallback(keras.callbacks.Callback): 

    def __init__(self, path_prefix): 
        super().__init__() 
        self.path_prefix = path_prefix # PATH_PREFIX ALSO CONTAINS THE SLASH, IS A STRING 

    def on_epoch_end(self, epoch, logs): 
        self.model.save(self.path_prefix+"epoch_"+str(epoch)+".keras")

def get_valid_sampler(k, i): 
    return lambda indices: [ indices[ini] for ini in range( math.floor(len(indices)*i/k) , len(indices) if i==k-1 else math.floor((len(indices)*(i+1))/k) ) ] 

def get_valid_samplers(k=5): # TODO: MAKE THE SAMPLERS 
    samplers = [] 
    for i in range(k): 
        '''def sampler(indices): 
            if i==0: lower = 0 
            else: lower = math.floor(len(indices)*i/k) 

            if i == k-1: upper = len(indices) 
            else: upper = math.floor((len(indices)*(i+1))/k)

            return [ indices[ini] for ini in range(lower, upper) ] '''
        
        samplers.append(get_valid_sampler(k, i)) 

        #del sampler 
    return samplers 



