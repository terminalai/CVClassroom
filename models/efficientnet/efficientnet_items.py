# modified from https://github.com/calmiLovesAI/EfficientNet_TensorFlow2/blob/master/efficientnet.py 


import tensorflow as tf 
import keras 
import math 



def swish(x):
    return x * tf.nn.sigmoid(x)


def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same", kernel_initializer=conv_kernel_initializer)
        self.expand_conv = keras.layers.Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same", kernel_initializer=conv_kernel_initializer)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output


import numpy as np 
import tensorflow.compat.v1 as tfv1 
def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    """Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.

    Args:
        shape: shape of variable
        dtype: dtype of variable
        partition_info: unused

    Returns:
        an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tfv1.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


class MBConv(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same", kernel_initializer=conv_kernel_initializer)
        self.bn1 = keras.layers.BatchNormalization()
        self.dwconv = keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same", depthwise_initializer=conv_kernel_initializer)
        self.bn2 = keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same", kernel_initializer=conv_kernel_initializer)
        self.bn3 = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = keras.layers.add([x, inputs])
        return x



def get_mbconv_block_layers(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    block = [] 
    for i in range(layers):
        if i == 0:
            block.append(MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
        else:
            block.append(MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
    return block






class EfficientNetItems(): 

    @staticmethod 
    def print(*args, **kwargs): 
        pass 
        #print(*args, **kwargs) 

    @staticmethod 
    def get_conv1(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('conv1')
        return keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                                                kernel_size=(3, 3),
                                                strides=2,
                                                padding="same", kernel_initializer=conv_kernel_initializer)

    @staticmethod 
    def get_bn1(): 
        EfficientNetItems.print('bn1')
        return keras.layers.BatchNormalization() 


    @staticmethod 
    def get_block1(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('block1') 
        return get_mbconv_block_layers(in_channels=round_filters(32, width_coefficient),
                                            out_channels=round_filters(16, width_coefficient),
                                            layers=round_repeats(1, depth_coefficient),
                                            stride=1,
                                            expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)

    @staticmethod 
    def get_block2(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('block2') 
        return get_mbconv_block_layers(in_channels=round_filters(16, width_coefficient),
                                            out_channels=round_filters(24, width_coefficient),
                                            layers=round_repeats(2, depth_coefficient),
                                            stride=2,
                                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)

    @staticmethod 
    def get_block3(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('block3') 
        return get_mbconv_block_layers(in_channels=round_filters(24, width_coefficient),
                                            out_channels=round_filters(40, width_coefficient),
                                            layers=round_repeats(2, depth_coefficient),
                                            stride=2,
                                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)

    @staticmethod 
    def get_block4(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('block4') 
        return get_mbconv_block_layers(in_channels=round_filters(40, width_coefficient),
                                            out_channels=round_filters(80, width_coefficient),
                                            layers=round_repeats(3, depth_coefficient),
                                            stride=2,
                                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)

    @staticmethod 
    def get_block5(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('block5') 
        return get_mbconv_block_layers(in_channels=round_filters(80, width_coefficient),
                                            out_channels=round_filters(112, width_coefficient),
                                            layers=round_repeats(3, depth_coefficient),
                                            stride=1,
                                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)

    @staticmethod 
    def get_block6(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('block6') 
        return get_mbconv_block_layers(in_channels=round_filters(112, width_coefficient),
                                            out_channels=round_filters(192, width_coefficient),
                                            layers=round_repeats(4, depth_coefficient),
                                            stride=2,
                                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)

    @staticmethod 
    def get_block7(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('block7') 
        return get_mbconv_block_layers(in_channels=round_filters(192, width_coefficient),
                                            out_channels=round_filters(320, width_coefficient),
                                            layers=round_repeats(1, depth_coefficient),
                                            stride=1,
                                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)

    @staticmethod 
    def get_conv2(depth_coefficient, width_coefficient, drop_connect_rate): 
        EfficientNetItems.print('conv2') 
        return keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same", kernel_initializer=conv_kernel_initializer)

    @staticmethod 
    def get_bn2(): 
        EfficientNetItems.print('bn2') 
        return keras.layers.BatchNormalization()

    @staticmethod 
    def get_pool(): 
        EfficientNetItems.print('pool') 
        return keras.layers.GlobalAveragePooling2D()

    @staticmethod 
    def get_dropout(dropout_rate): 
        EfficientNetItems.print('dropout') 
        return keras.layers.Dropout(rate=dropout_rate)

    @staticmethod 
    def get_fc(NUM_CLASSES:int): 
        EfficientNetItems.print('fc') 
        return keras.layers.Dense(units=NUM_CLASSES, activation=keras.activations.softmax)



