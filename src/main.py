#!/usr/bin/env python
# coding: utf-8


import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from data_prep import prepare_mnist, prepare_imagenet, create_val_img_folder
from svm import SVM, MultiClassHingeLoss
from alexnet import AlexNet


class Args(object):
      def __init__(self, batch_size=64, test_batch_size=1000, epochs=10, lr=0.01, 
                   optimizer='adam', momentum=0.5, seed=1, log_interval=100, 
                   dataset='mnist', data_dir=os.getcwd(), cuda=True, model='SVM', 
                   features=784, classes=10, reg=False, margin=1, topk=5, 
                   results=os.path.join(os.getcwd(), 'results')):
        
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
        self.results = results


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

    # Images don't show on Ubuntu
    # plt.tight_layout()

    # Save results
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    filename = args.dataset + '_' + args.model + '_plot.png'
    fig.savefig(os.path.join(args.results, filename))


if __name__ == '__main__':
    run_experiment(Args(topk=1, epochs=2))
    run_experiment(Args(dataset='tiny-imagenet-200', batch_size=1000, features=12288, classes=200, margin=20, epochs=3))
    run_experiment(Args(dataset='tiny-imagenet-200', model='AlexNet', batch_size=1000, classes=200))


# TODO
# Mnist - SVM - Top1
# run_experiment(Args(topk=1))

# Tiny Imagenet - SVM - Top5
# run_experiment(Args(dataset='tiny-imagenet-200', batch_size=1000, features=12288, classes=200, margin=20))

# Tiny Imagenet - AlexNet - Top5
# run_experiment(Args(dataset='tiny-imagenet-200', model='AlexNet', batch_size=1000, classes=200))

# parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')
# args = parser.parse_args()
