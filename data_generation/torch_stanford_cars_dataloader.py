
import os
import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms as tt 
from torchvision.datasets import ImageFolder 

import matplotlib.pyplot as plt 
from torchvision.utils import make_grid 

DATA_DIR_TRAIN = './stanford-car-dataset-by-classes-folder/car_data/car_data/train'
train_classes = os.listdir(DATA_DIR_TRAIN)

DATA_DIR_TEST = './stanford-car-dataset-by-classes-folder/car_data/car_data/test'
test_classes = os.listdir(DATA_DIR_TEST)

#print("CLASSES:",train_classes[:5], test_classes[:5]) 


def find_train_classes(dir):
    train_classes = os.listdir(dir)
    train_classes.sort()
    train_class_to_idx = {train_classes[i]: i for i in range(len(train_classes))}
    return train_classes, train_class_to_idx
train_classes, train_c_to_idx = find_train_classes(DATA_DIR_TRAIN) 


def find_test_classes(dir):
    test_classes = os.listdir(dir)
    test_classes.sort()
    test_class_to_idx = {test_classes[i]: i for i in range(len(test_classes))}
    return test_classes, test_class_to_idx
test_classes, test_c_to_idx = find_test_classes(DATA_DIR_TEST)


#print("CLASSES:",train_classes[:5], test_classes[:5]) 


# For data transforms (normalization & data augmentation)
normalize_stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def get_datasets(size=(256, 256)): 

    train_tfms = tt.Compose([tt.Resize(size),
                             tt.RandomRotation(0),
                             tt.ToTensor(),
                             tt.Normalize(*normalize_stats, inplace = True)])
    valid_tfms = tt.Compose([tt.Resize(size),
                            tt.ToTensor(),
                            tt.Normalize(*normalize_stats)])

    train_ds = ImageFolder(DATA_DIR_TRAIN, train_tfms)
    valid_ds = ImageFolder(DATA_DIR_TEST, valid_tfms)

    return train_ds, valid_ds 

def get_dataloaders(size=(256, 256), batch_size = 128): 

    train_ds, valid_ds = get_datasets(size) 

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

    return train_dl, valid_dl 


# TO USE:
# tl, vl = get_dataloaders()
# tf, tl = next(iter(tl))



# for visualization purposes 

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, *normalize_stats)
        ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
        plt.show()
        break 


# testing 
if __name__ == "__main__": 
    tl, vl = get_dataloaders()
    tf, tl = next(iter(tl)) 
    print(len(tf), tf[0].shape, len(tl), tl[0]) 
    show_batch(vl) 

