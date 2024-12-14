import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import cv2

# Gaussian noise: Adding a random normal distribution to the image's intensity values
def add_gaussian_noise(img):
    # Get gray scale image
    img_gray = img[:,:,1]

    # Create normal distribution of the same size as the image
    noise = np.random.normal(0,50,img.shape)

    # Add noise
    img_noised = img_gray + noise

    # Clip pixel value to 255
    return np.clip(img_noised,0,255).astype(np.uint8)

# Apparently there is already a Gaussian noise layer in keras
import tensorflow.keras.layers import GaussianNoise

def add_gaussian_blur(domain=2.1, mean=0.1, variance=0.2):
    # Gettting Gaussian blur kernel
    X = np.arange(-domain+mean, domain+mean, variance)
    Y = np.arange(-domain+mean, domain+mean, variance)
    X,Y = np.meshgrid(X,Y)
    R = np.sqrt(X**2+Y**2)
    kernel = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5*R**2)) # default return 21x21 kernel of gaussian blur filter
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    

# Simple image augmentation
def add_augmentation(img, save_to_dir='test_folder', save_prefix='aug', save_format='png', img_size=128):
    augmented = []
    datagen = ImageDataGenerator(
        rotation_range=45, # Rotation
        width_shift_range=0.2, # Horizontal shift
        height_shift_range=0.2, # Vertical shift
        shear_range=0.2,
        zoom_range=0.2, # Zoom
        horizontal_flip=True,
        fill_mode='wrap', cval=125)
    x = img.reshape((1,)+img.shape)
    i=0
    for batch in datagen.flow(x, batch_size=16,
                                save_to_dir=save_dir,
                                save_prefix=save_prefix,
                                save_format=save_format):
        i+=1
        if i>20:
            break
    img_dir = save_to_dir + '/'
    imgs = os.listdir(img_dir)
    # Save the images to numpy list
    for i, img_name in enumerate(imgs):
        if img_name.split('.')[1] == 'png':
            img = io.imread(img_dir+img_name)
            img = Image.fromarray(img,'RGB')
            img = img.resize((img_size,img_size))
            augmented.append(np.array(img))
    # Return a larger numpy array of the arrays of images
    return np.array(augmented)
