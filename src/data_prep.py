#!/usr/bin/env python
# coding: utf-8


import os
import torch
from torchvision import datasets
from torchvision import transforms
import torchsample.transforms as tstf


def prepare_mnist(args):
    DatasetClass = datasets.MNIST
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    
    train_dataset = DatasetClass(dataset_dir, train=True, download=True, 
                                 transform=transforms.Compose([
                                     transforms.ToTensor(), 
                                     transforms.Normalize((0.1307,), (0.3081,))]))
    
    test_dataset = DatasetClass(dataset_dir, train=False, 
                                transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))]))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=args.test_batch_size, 
                                              shuffle=True, **kwargs)
    
    return train_loader, test_loader, train_dataset, test_dataset


def prepare_imagenet(args):
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val', 'images')
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Pre-calculated mean & std on imagenet:
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # For other datasets, we could just simply use 0.5:
    # norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    print('Preparing dataset ...')
    # Normalization
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        if args.pretrained else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Normal transformation
    if args.pretrained:
        train_trans = [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), 
                        transforms.ToTensor(), norm]
        val_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
    else:
        train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm]
        val_trans = [transforms.ToTensor(), norm]

    # Data augmentation (torchsample)
    if args.data_augmentation:
        train_trans += [tstf.RandomGamma(0.5, 1.5)]

    train_data = datasets.ImageFolder(train_dir, 
                                    transform=transforms.Compose(train_trans))
    
    val_data = datasets.ImageFolder(val_dir, 
                                    transform=transforms.Compose(val_trans))
    
    print('Preparing data loaders ...')
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                                    shuffle=True, **kwargs)
    
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, 
                                                  shuffle=True, **kwargs)
    
    return train_data_loader, val_data_loader, train_data, val_data


def create_val_img_folder(args):
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))
