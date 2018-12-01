#!/usr/bin/env python
# coding: utf-8

# # 1. Preparation

# In[ ]:


# If on Google Colab:

# Install pytorch and tqdm (if necessary)
#!pip install torch
#!pip install torchvision
#!pip install tqdm

# Mount your google drive as the data drive
# This will require google authorization
#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter


# In[ ]:


class Args(object):
      def __init__(self, batch_size=64, test_batch_size=1000, epochs=10, lr=0.01, 
                   optimizer='adam', momentum=0.5, seed=1, log_interval=100, 
                   dataset='mnist', data_dir='./', cuda=True, model='SVM', 
                   features=784, classes=10, reg=False, margin=1, topk=5):
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.momentum = momentum
        self.seed = seed
        self.log_interval = log_interval
        self.dataset = dataset
        self.data_dir = data_dir # Path to datasets
        self.cuda = cuda and torch.cuda.is_available()
        self.model = model
        self.features = features # Number of input features
        self.classes = classes # Number of classes
        self.reg = reg # L2 regularization
        self.margin = margin # Margin in hinge loss
        self.topk = topk # Top-k accuracy


# In[ ]:


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


# In[ ]:


def prepare_imagenet(args):
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val/images')
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    
    print('Preparing dataset ...')
    train_data = datasets.ImageFolder(train_dir, 
                                      transform=transforms.Compose([
                                          #transforms.RandomResizedCrop(56),
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()]))
    
    val_data = datasets.ImageFolder(val_dir, 
                                    transform=transforms.Compose([
                                        #transforms.RandomResizedCrop(56),
                                        transforms.ToTensor()]))
    
    print('Preparing data loaders ...')
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                                    shuffle=True, **kwargs)
    
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, 
                                                  shuffle=True, **kwargs)
    
    return train_data_loader, val_data_loader, train_data, val_data


# Pre-calculated mean & std on imagenet:
# ```
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ```
# 
# For other datasets, we could just simply use 0.5:
# ```
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ```

# In[ ]:


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


# # 2. Models

# In[ ]:


class SVM(nn.Module):
    def __init__(self, n_feature, n_class):
        super(SVM, self).__init__()
        self.fc = nn.Linear(n_feature, n_class)

    def forward(self, x):
        x = self.fc(x)
        return x


# In[ ]:


class MultiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=20, size_average=True):
        super(MultiClassHingeLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.size_average = size_average
        
    def forward(self, output, y):
        output_y = output[torch.arange(0, y.size()[0]).long(), y.data].view(-1, 1)
        loss = output - output_y + self.margin
        loss[torch.arange(0, y.size()[0]).long(), y.data] = 0
        loss[loss<0] = 0
        
        if(self.p != 1):
            loss = torch.pow(loss, self.p)
        
        loss = loss.mean() if self.size_average else loss.sum()
        return loss


# In[ ]:


class AlexNet(nn.Module):
    def __init__(self, n_class):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=6, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_class),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 8 * 8)
        x = self.classifier(x)
        return x


# # 3. Train & Test

# In[ ]:


def train(model, criterion, optimizer, train_loader, epoch, 
          total_minibatch_count, train_losses, train_accs, args):
    model.train()
    correct, total_loss, total_acc = 0., 0., 0.
    progress_bar = tqdm.tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Stretch images to a 1D vector
        if args.model == 'SVM':
            data = data.view(-1, args.features)
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # L2 regularization
        if args.reg:
            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
            loss += 1/2 * l2_reg
        
        # Backpropagation  
        loss.backward()
        optimizer.step()
        
        # Compute top-k accuracy
        top_indices = torch.topk(output.data, args.topk)[1].t()
        match = top_indices.eq(target.view(1, -1).expand_as(top_indices))
        accuracy = match.view(-1).float().mean() * args.topk
        correct += match.view(-1).float().sum(0)

        if args.log_interval != 0 and total_minibatch_count % args.log_interval == 0:
            train_losses.append(loss.data[0])
            train_accs.append(accuracy.data[0])
            
        total_loss += loss.data
        total_acc += accuracy.data
            
        progress_bar.set_description(
            'Epoch: {} loss: {:.4f}, acc: {:.2f}'.format(
                epoch, total_loss / (batch_idx + 1), total_acc / (batch_idx + 1)))

        total_minibatch_count += 1

    return total_minibatch_count


# In[ ]:


def test(model, criterion, test_loader, epoch, val_losses, val_accs, args):
    model.eval()
    test_loss, correct = 0., 0.
    progress_bar = tqdm.tqdm(test_loader, desc='Validation')
    
    with torch.no_grad():
        for data, target in progress_bar:
            if args.model == 'SVM':
                data = data.view(-1, args.features)
            
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            output = model(data)
            test_loss += criterion(output, target).data
            
            top_indices = torch.topk(output.data, args.topk)[1].t()
            match = top_indices.eq(target.view(1, -1).expand_as(top_indices))
            correct += match.view(-1).float().sum(0)
            
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    val_losses.append(test_loss)
    val_accs.append(acc)
    
    progress_bar.clear()
    progress_bar.write(
        '\nEpoch: {} validation test results - Average val_loss: {:.4f}, val_acc: {}/{} ({:.2f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return acc


# In[ ]:


def run_experiment(args):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Dataset
    if args.dataset == 'mnist':
        train_loader, test_loader, _, _ = prepare_mnist(args)
    else:
        # create_val_img_folder(args)
        train_loader, test_loader, _, _ = prepare_imagenet(args)
    
    # Model & Criterion
    if args.model == 'AlexNet':
        model = AlexNet(args.classes)
        criterion = nn.CrossEntropyLoss(size_average=False)
    else:
        model = SVM(args.features, args.classes)
        criterion = MultiClassHingeLoss(margin=args.margin, size_average=False)
    if args.cuda:
        model.cuda()
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    total_minibatch_count = 0
    val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        total_minibatch_count = train(model, criterion, optimizer, train_loader, 
                                      epoch, total_minibatch_count, train_losses, 
                                      train_accs, args)
        
        val_acc = test(model, criterion, test_loader, epoch, val_losses, val_accs, args)
        
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    axes[0].plot(train_losses)
    axes[0].set_title('Loss')
    axes[1].plot(train_accs)
    axes[1].set_title('Acc')
    axes[1].set_ylim([0, 1])
    axes[2].plot(val_losses)
    axes[2].set_title('Val loss')
    axes[3].plot(val_accs)
    axes[3].set_title('Val Acc')
    axes[3].set_ylim([0, 1])
    plt.tight_layout()


# # 4. Experiments

# In[ ]:


# Mnist - SVM - Top1
run_experiment(Args(topk=1))


# In[ ]:


# Tiny Imagenet - SVM - Top5
# run_experiment(Args(dataset='tiny-imagenet-200', batch_size=1000, features=12288, classes=200, margin=20))


# In[ ]:


# Tiny Imagenet - AlexNet - Top5
# run_experiment(Args(dataset='tiny-imagenet-200', model='AlexNet', batch_size=1000, classes=200))

