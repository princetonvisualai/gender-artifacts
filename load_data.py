import pickle
import glob
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import transform

class Dataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=T.ToTensor()):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        ID = self.img_paths[index]
        img = Image.open(ID).convert('RGB')
        X = self.transform(img)
        y = self.img_labels[ID]

        return X, y, ID

def create_dataset(labels_path, B=200, train=True, res=224):

    img_labels = pickle.load(open(labels_path, 'rb'))
    img_paths = sorted(list(img_labels.keys()))

    # Common from here
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        random_resize = True
        if random_resize:
            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.Resize(res),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(224),
                T.Resize(res),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        shuffle = True
    else:
        center_crop = True
        transform = T.Compose([
               T.Resize(256),
               T.CenterCrop(224),
               #T.Resize(res),
               T.ToTensor(),
               normalize
        ])

        shuffle = False

    dset = Dataset(img_paths, img_labels, transform)
    loader = DataLoader(dset, batch_size=B, shuffle=shuffle, num_workers=1)
    return loader
