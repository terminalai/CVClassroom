import tensorflow as tf 
import keras 
import math 

#tf.debugging.set_log_device_placement(True)

warn = True 

def quality_matrix(k, alpha=0.3):
    """r
    Quality matrix Q. Described in the eq (17) so that eps = QX, where X is the input. 
    Alpha is 0.3, as mentioned in Appendix D.
    """
    identity = tf.linalg.diag(tf.ones(k))
    shift_identity = tf.zeros(k, k) 
    for i in range(k):
        shift_identity[(i+1)%k, i] = 1
    opt = -alpha * identity + alpha * shift_identity
    return opt

def optimal_quality_matrix(k):
    """r
    Optimal Quality matrix Q. Described in the eq (19) so that eps = QX, where X is the input. 
    Suppose 1_(kxk) is tf.ones
    """
    return tf.linalg.diag(tf.ones(k)) * -k/(k+1) + tf.ones((k, k)) / (k+1)




def get_linear_transform_noise_of(x:tf.Tensor, noise_mat=None, ): 
    #assert len(x.shape)==2, "x of LTNL must be batched one-dimensional data. "
    if (x.dtype != tf.float32): 
        if warn: 
            print("\nWARNING: DType of linear transform noise input is not float32. Attempting typecast (no error means success)\n")
        x = tf.cast(x, tf.float32) 

    #if (noise_mat != None): assert len(noise_mat.shape)==2, "Noise matrix must be a 2-dimensional matrix" 

    if (noise_mat==None): noise_mat = optimal_quality_matrix(x.shape[-1]) # choose optimal quality matrix by default 
    #print("ABOUT TO MATRIX MULITPLY")    
    return x@noise_mat # matrix multiply them 

def add_linear_transform_noise_to(x:tf.Tensor, noise_mat=None, ): 
    #assert (len(x.shape)==2), "SHAPE OF X MUST BE 2 TO ADD LTN"

    '''if (len(x.shape) == 3): 
        # noise to each colour channel instead 
        return 

        x = tf.transpose(x, perm=(2,0,1)).view(shape[0],shape[2], math.sqrt(shape[1]), math.sqrt(shape[1]))
    
    print("RESIZED:",x.shape)'''

    return get_linear_transform_noise_of(x, noise_mat) + x 

    '''print("GOT LTN YAY")
    
    if (shape is not None): 
        x = x.view(shape[0], shape[1], -1).permute(0,2,1) 

    return x '''


class LinearTransformNoiseLayer(keras.layers.Layer): 
    noisetype_names = ['ltnl', 'linear_transform']

    def __init__(self, noise_mat=None, batched=True): 
        super(LinearTransformNoiseLayer, self).__init__() 
        if (noise_mat is not None): assert len(noise_mat.shape)==2, "Noise matrix must be a 2-dimensional matrix" 
        self.noise_mat = noise_mat 
        self.batched = batched 

    def build(self, input_shape): 
        second_last = False 
        if (self.batched and (len(input_shape)==4)): second_last=True 
        if ((not self.batched) and (len(input_shape)==3)): second_last=True
        self.last_is_channels = second_last 
        if (self.noise_mat == None): 
            self.noise_mat = optimal_quality_matrix(input_shape[-1-int(second_last)]) 

    '''def get_output_2d(self, x): 
        return add_linear_transform_noise_to(x, self.noise_mat) 
    
    def get_output(self, x): 
        #print("ADDING NOISE TO:", x) 
        if len(x.shape)==2: 
            return self.get_output_2d(x) 
        elif len(x.shape)==3: 
            return tf.transpose(tf.map_fn(self.get_output_2d, tf.transpose(x, perm=[2,0,1])), perm=[1,2,0])''' 

    def call(self, input): 
        if self.last_is_channels:
            lens = len(input.shape) 
            input = tf.transpose(input, perm=[lens-1, *range(lens-1)])
        
        res = add_linear_transform_noise_to(input) 

        if (lens is not None): 
            res = tf.transpose(res, perm=[*range(1,lens), 0])
        
        return res 

        '''if self.batched: 
            # batched data 
            return tf.map_fn(self.get_output, input)
        else: 
            return self.get_output(input)''' 
    
    def compute_output_shape(self, input_shape): 
        if (self.batched) and ((2 <= len(input_shape)) and (len(input_shape) <= 4)): 
            return input_shape 
        elif (not self.batched) and ((1 <= len(input_shape)) or (len(input_shape) <= 3)): 
            return input_shape 
        else: 
            return 0 
    



