import sys
if 'idlelib.run' in sys.modules: sys.path.pop(1) # fixes an issue with import 
sys.path.insert(1, './tests/')
sys.path.insert(1, '.')

import os
os.environ['KERAS_BACKEND'] = "tensorflow" 


import tensorflow as tf 
import keras 

import numpy as np 

from data_generation import StanfordCarsDataloader as SCDL
from model_noise import LinearTransformNoiseLayer as LTNL 
from models.efficientnet.efficientnet_items import * 

import utils 

tf.config.list_physical_devices('GPU')

num_epochs = 12 
valid_epochs = 3 

metrics = [keras.metrics.Accuracy(), keras.metrics.TopKCategoricalAccuracy(k=5)] 


dnn_params = {
    'depth_coefficient': 1.0, 
    'width_coefficient': 1.0, 
    'drop_connect_rate': 0.2, 
}
dropout_rate = 0.2 

train_batch_size = 32 
valid_batch_size = 16 

# for cross-validation 
valid_samplers = utils.get_valid_samplers(k=5)

default_target_img_shape = (224, 224, 3) 


def get_noiseless_model(n_classes, target_img_shape=default_target_img_shape): 
    return keras.Sequential([
        keras.layers.Input(target_img_shape),

        EfficientNetItems.get_conv1(**dnn_params), 
        EfficientNetItems.get_bn1(), 
        keras.layers.Activation(keras.activations.swish), 

        *EfficientNetItems.get_block1(**dnn_params), 
        *EfficientNetItems.get_block2(**dnn_params), 
        *EfficientNetItems.get_block3(**dnn_params), 
        *EfficientNetItems.get_block4(**dnn_params), 
        *EfficientNetItems.get_block5(**dnn_params), 
        *EfficientNetItems.get_block6(**dnn_params), 
        *EfficientNetItems.get_block7(**dnn_params), 

        EfficientNetItems.get_conv2(**dnn_params), 
        EfficientNetItems.get_bn2(), 
        keras.layers.Activation(keras.activations.swish), 
        EfficientNetItems.get_pool(), 
        EfficientNetItems.get_dropout(dropout_rate), 
        EfficientNetItems.get_fc(n_classes), 
    
    ])

def get_ltnl_model(n_classes, target_img_shape=default_target_img_shape): 
    return keras.Sequential([
        keras.layers.Input(target_img_shape),

        EfficientNetItems.get_conv1(**dnn_params), 
        EfficientNetItems.get_bn1(), 
        keras.layers.Activation(keras.activations.swish), 

        LTNL(), 

        *EfficientNetItems.get_block1(**dnn_params), 
        LTNL(), 
        *EfficientNetItems.get_block2(**dnn_params), 
        LTNL(), 
        *EfficientNetItems.get_block3(**dnn_params), 
        LTNL(), 
        *EfficientNetItems.get_block4(**dnn_params), 
        LTNL(), 
        *EfficientNetItems.get_block5(**dnn_params), 
        LTNL(), 
        *EfficientNetItems.get_block6(**dnn_params), 
        LTNL(), 
        *EfficientNetItems.get_block7(**dnn_params), 
        LTNL(), 

        EfficientNetItems.get_conv2(**dnn_params), 
        EfficientNetItems.get_bn2(), 
        keras.layers.Activation(keras.activations.swish), 
        LTNL(), 
        EfficientNetItems.get_pool(), 
        EfficientNetItems.get_dropout(dropout_rate), 
        EfficientNetItems.get_fc(n_classes), 
    ])





def train_noiseless_models(): 
    print("EfficientNetB0 (no noise injected) MODEL TESTING: ")
    
    
    # do cross-validation 
    for vsidx in range(len(valid_samplers)): 

        save_dir = "./NoiselessB0_valid_"+str(vsidx)+"/" 
        try: 
            os.mkdir(save_dir) 
        except: 
            pass # just means it's already been made 


        noiseless_model = get_noiseless_model() 
        #noiseless_model.summary() 
        noiseless_model.compile(optimizer='adam',
                    loss = keras.losses.BinaryCrossentropy(),#from_logits=True),
                    metrics = metrics) 

        valid_sampler = valid_samplers[vsidx] 
        train_dataloader = SCDL('train', data_shape=default_target_img_shape, valid_sampler=valid_sampler, batch_size=train_batch_size) 
        valid_dataloader = SCDL('valid', data_shape=default_target_img_shape, valid_sampler=valid_sampler, batch_size=valid_batch_size) 

        noiseless_model.fit(x=train_dataloader, validation_data=valid_dataloader, 
                            verbose=2, epochs=num_epochs, validation_freq=valid_epochs, 
                            callbacks = [utils.SaveEveryEpochCallback(save_dir)]) 

def train_ltnl_models(): 
    print("EfficientNetB0 + LTNL MODEL TESTING: ")
    
    
    # do cross-validation 
    for vsidx in range(len(valid_samplers)): 

        save_dir = "./LTNLB0_valid_"+str(vsidx)+"/" 
        try: 
            os.mkdir(save_dir) 
        except: 
            pass # just means it's already been made 


        ltnl_model = get_ltnl_model(10) 
        #ltnl_model.summary() 
        ltnl_model.compile(optimizer='adam',
                    loss = keras.losses.BinaryCrossentropy(),#from_logits=True),
                    metrics = metrics) 

        valid_sampler = valid_samplers[vsidx] 
        train_dataloader = SCDL('train', data_shape=default_target_img_shape, valid_sampler=valid_sampler, batch_size=train_batch_size) 
        valid_dataloader = SCDL('valid', data_shape=default_target_img_shape, valid_sampler=valid_sampler, batch_size=valid_batch_size) 

        ltnl_model.fit(x=train_dataloader, validation_data=valid_dataloader, 
                            verbose=2, epochs=num_epochs, validation_freq=valid_epochs, 
                            callbacks = [utils.SaveEveryEpochCallback(save_dir)]) 



if __name__ == "__main__": 

    ltnl_model = get_ltnl_model(10) 
    #ltnl_model.summary() 
    ltnl_model.compile(optimizer='adam',
                loss = keras.losses.BinaryCrossentropy(),#from_logits=True),
                metrics = metrics) 

    valid_sampler = valid_samplers[0] 
    train_dataloader = SCDL('train', data_shape=default_target_img_shape, valid_sampler=valid_sampler, batch_size=1) 
    x, y = train_dataloader[0] 
    print("X SHAPE", x.shape)
    print("Y SHAPE", y)

    print("OUTPUT", ltnl_model.predict(x)) 

    1/0 
    train_noiseless_models() 
    train_ltnl_models()



