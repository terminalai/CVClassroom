import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import cv2
import tempfile 

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



def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    credits: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy 
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

gaussian_kernels = {}#{(21, 1.0): gkern(21, 1.0)}

# input/output format: cv2 image 
def cv2_add_gaussian_blur(img, l=21, std=1.0): 
    if (l, std) not in gaussian_kernels.keys(): 
        gaussian_kernels[(l,std)] = gkern(l,std) 
    return cv2.filter2D(src=img, ddepth=-1, kernel = gaussian_kernels[(l, std)])

# input/ouptut format: PIL image 
def add_gaussian_blur(img, l=21, std=1.0): 
    if (l, std) not in gaussian_kernels.keys(): 
        gaussian_kernels[(l,std)] = gkern(l,std) 
    
    cv2_img = np.array(img)
    return Image.fromarray(np.array(cv2.filter2D(src=cv2_img, ddepth=-1, kernel = gaussian_kernels[(l, std)]))) 
# we arent using PIL's ImageFilter because it's kernel sizes aren't very flexible 



# Simple image augmentation
def add_augmentation(img, img_size=(224, 224), num=2): # INPUT FORMAT: PIL image 

    img = cv2.cvtColor(np.array(img) , cv2.COLOR_BGR2RGB) 

    augmented = [] # Output format: PIL image 
    datagen = ImageDataGenerator(
        rotation_range=45, # Rotation
        width_shift_range=0.2, # Horizontal shift
        height_shift_range=0.2, # Vertical shift
        shear_range=0.2,
        zoom_range=0.2, # Zoom
        horizontal_flip=True,
        fill_mode='wrap', cval=125)
    
    x = img.reshape((1,)+img.shape)


    with tempfile.TemporaryDirectory() as tmpdir: 
        # make num images 
        i=0
        for batch in datagen.flow(x, batch_size=16,
                                    save_to_dir=tmpdir,
                                    save_prefix='aug',
                                    save_format='png'):
            i+=1
            if i>num:
                break
        
        img_dir = tmpdir + '/'
        imgs = os.listdir(img_dir)
        # Save the images to numpy list
        for i, img_name in enumerate(imgs):
            if img_name.split('.')[1] == 'png':
                img = cv2.imread(img_dir+img_name)
                img = Image.fromarray(img,'RGB')
                img = img.resize(img_size)
                augmented.append(Image.fromarray(np.array(img))) 
    
    # Return a list of the numpy arrays of images
    return augmented #np.array(augmented)
