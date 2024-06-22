import keras 
import models.efficientnet.efficientnet_b0 as efficientnet_b0 
from noise_layers import LinearTransformNoiseLayer as LTNL 

import utils 

import sys
if 'idlelib.run' in sys.modules: sys.path.pop(1) # fixes an issue with import 
sys.path.insert(1, '../')

import os
os.environ['KERAS_BACKEND'] = "tensorflow" 



default_target_img_shape = (224, 224, 3) 

class EfficientNetModel(): # each expert can be an EfficientNetModel 
    def __init__(self, save_dir, n_classes, model_specs='B0', img_shape=default_target_img_shape, noise_type='ltnl', ): 
        assert model_specs in ['B0'], "Model not available yet" 
        assert noise_type in [None, *LTNL.noisetype_names], "Noise type not available yet" 

        try: os.mkdir(save_dir) 
        except: pass # means it's already made before 

        self.n_classes = n_classes 
        self.model_specs = model_specs 
        self.img_shape = img_shape 
        self.noise_type = noise_type 
        self.save_dir = save_dir 
        
        self.reset() 


    def reset(self): 
        if self.model_specs=='B0': 
            if self.noise_type == None: 
                self.model = efficientnet_b0.get_noiseless_model(self.n_classes, self.img_shape) 
            elif self.noise_type in LTNL.noisetype_names: 
                self.model = efficientnet_b0.get_ltnl_model(self.n_classes, self.img_shape) 
        
        self.trained_already = False 
    

    def train(self, train_dataloader, valid_dataloader, optimizer='AdamW', 
              loss=keras.losses.BinaryCrossentropy(), # from_logits=False, as SoftMax activation is assumed ( https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow ) 
              metrics=[keras.metrics.Accuracy(), keras.metrics.TopKCategoricalAccuracy(k=5)], 
              num_epochs=12, valid_freq=3, callbacks:list = None, 
              compile_kwargs={}, **fit_kwargs, ): 
        
        assert (not self.trained_already), "Model has already been trained." 

        if callbacks == None: 
            [ utils.SaveEveryEpochCallback(self.save_dir) ] 

        self.model.compile(optimizer=optimizer, loss=loss, metrics = metrics, **compile_kwargs) 

        self.model.fit(x=train_dataloader, validation_data=valid_dataloader, 
                       verbose=2, epochs=num_epochs, validation_freq=valid_freq, 
                       callbacks=callbacks, **fit_kwargs) 


        self.trained_already = True 
    
    def test(self, test_dataloader, metrics=[keras.metrics.Accuracy(), keras.metrics.TopKCategoricalAccuracy(k=5)], ): 
        pass # TODO: TEST THE MODEL!!! 

    def predict(self, x, *args, **kwargs): 
        return self.model.predict(x, *args, **kwargs) 







