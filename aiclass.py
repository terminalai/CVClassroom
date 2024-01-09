from keras.models import Sequential 
from keras.layers import Dense, Conv2D 
from keras.callbacks import LambdaCallback, ModelCheckpoint 
from keras.optimizers import AdamW 

import utils 
import params 

import os 

class Model_Conv2D_Dense(): 
    def __init__(self, conv2d_input_shape:tuple, conv2d_kwargss:list, 
                 dense_input_shape:tuple, dense_kwargss:list, 
                 save_checkpoint_dir:str, 
                 kernel_initializer="glorot_uniform", bias_initializer="zeros", 
                 loss="binary_crossentropy", optimizer=AdamW(learning_rate=0.001, weight_decay=0.0), metrics=["accuracy"]): 
        self.model = Sequential() 

        self.model.add(Conv2D(**conv2d_kwargss[0], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=conv2d_input_shape)) 
        for conv2d_kwargs in conv2d_kwargss[1:]: 
            self.model.add(Conv2D(**conv2d_kwargs, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)) 
        
        self.model.add(Dense(**dense_kwargss[0], kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=dense_input_shape)) 
        for dense_kwargs in dense_kwargss[1:]: 
            self.model.add(Dense(**dense_kwargs, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)) 
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics) 

        self.save_checkpoint_dir = save_checkpoint_dir 

        try: 
            os.mkdir(save_checkpoint_dir) 
        except: 
            pass # probably already created 
    
    def save(self, save_location): 
        self.model.save_weights(save_location) 
    
    def load(self, load_location): 
        self.model.load_weights(load_location) 

    default_epoch_count = 150 
    default_batch_size=10 
    
    def train(self, x, y, epochs:int=default_epoch_count, batch_size:int=default_batch_size, callbacks:list=[], **kwargs): 
        # input: train inputs, train target outputs, number of epochs, batch size of training, other keyword arguments 
        # output: history of loss values and metric values during training 

        num_batches = (50000+batch_size-1)//batch_size 

        callbacks.append(ModelCheckpoint(self.save_checkpoint_dir+"epoch_{epoch:04d}.ckpt", verbose=1, save_weights_only=True, save_freq=5*num_batches)) 

        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size, **kwargs, callbacks=callbacks) # results are here :) 

    def test(self, x, y): 
        # input: test inputs, test target outputs 
        # output: results gotten by evaluating model, with metrics 
        results = self.model.evaluate(x, y) 
        return results 
    
    def sample(self, x): 
        # input: inputs to sample output for 
        # output: sampled outputs 
        return self.model.predict(x) 

        


