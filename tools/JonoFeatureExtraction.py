import argparse
import os
import random
import shutil
import time
import warnings
import pickle
import torchvision
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from ModelOperationsFeatEx import Model_Operations
import random
random.seed(0)
from random import shuffle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


#SET UP
name_of_file_to_save = "test.hdf5"
csv_file_name = "imagenet_1000_val.csv"
direc_for_csv_file = "/net/ironman/scratch/datasets/ImageNet/ILSVRC_2012/"
batch_size_set = 256
workers_set = 8
model = models.resnet50(pretrained=True)
model = model.to('cuda')
criterion = nn.CrossEntropyLoss()


class ImageNetPytorch(Dataset):
    def __init__(self, csv_file, images_path, debug=False, shuffle_samples=False,
                transform=[transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
                ):
        with open(csv_file, 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        if debug:
            self.files = self.files[:1000]
        if shuffle_samples:
            shuffle(self.files)
        self.images_path = images_path
        self.transform = transforms.Compose(transform)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        jpeg_path, label = self.files[index].split(',')
        img = Image.open(self.images_path+jpeg_path).convert('RGB')
        x = self.transform(img)
        y = torch.as_tensor(int(label)-1, dtype=torch.int64)
        return (x,y,jpeg_path.split("/")[-1], jpeg_path.split("/")[-2])

best_acc1 = 0


def main():
    cudnn.benchmark = True

    # Data loading code
    val_loader = torch.utils.data.DataLoader(
        ImageNetPytorch(csv_file_name,
                        direc_for_csv_file),
        batch_size=batch_size_set, shuffle=False,
        num_workers=workers_set, pin_memory=True)

    validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    model.eval()
    modelObj = Model_Operations(model)

    hf = h5py.File(name_of_file_to_save, "w")
    
    
    current_class = "n01440764" #"testing class. Don't use"
    current_class_features=[]
    current_class_logits = []
    current_class_im_name = []
    
    with torch.no_grad():
        all_data={}

        for i, (images, target, img_name, gt_class) in enumerate(val_loader):
            target = target.to('cuda')
            images = images.cuda()
            features, Logits = modelObj(images)
            output = Logits
            
            for gt,im_name,feature,Logit in zip(gt_class, img_name,features.tolist(),Logits.tolist()):
                print('WORKS')
                if gt != current_class:
                    g = hf.create_group(current_class)
                    g.create_dataset('features', data=np.array(current_class_features))
                    g.create_dataset('avgpool', data=np.array(current_class_logits))
                    g.create_dataset('image_names', data=np.array(current_class_im_name, dtype=h5py.string_dtype(encoding='utf-8')))
                    
                    current_class = gt
                    current_class_features=[]
                    current_class_logits = []
                    current_class_im_name = []
                    
                    current_class_features.append(feature)
                    current_class_logits.append(Logit)
                    current_class_im_name.append(im_name)

                else:
                    current_class_features.append(feature)
                    current_class_logits.append(Logit)
                    current_class_im_name.append(im_name)
                
            if i % args.print_freq == 0:
                progress.display(i)
                   
        print('BREAK DONE')

if __name__ == '__main__':
    main()
