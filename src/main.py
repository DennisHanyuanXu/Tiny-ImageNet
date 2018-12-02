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


parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')
parser.add_argument('--batch-size', default=1000, type=int, 
                    help='mini-batch size for training (default: 1000)')
parser.add_argument('--test-batch-size', default=1000, type=int, 
                    help='mini-batch size for testing (default: 1000)')
parser.add_argument('--epochs', default=10, type=int, 
                    help='number of total epochs to run (default: 10)')
parser.add_argument('--lr', default=0.01, type=float, 
                    help='learning rate (default: 0.01)')
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], 
                    help='optimizer (default: adam)')
parser.add_argument('--momentum', default=0.5, type=float, 
                    help='momentum (default: 0.5)')
parser.add_argument('--seed', default=1, type=int, 
                    help='seed for initializing training (default: 1)')
parser.add_argument('--log-interval', default=100, type=int, 
                    help='batches to wait before logging detailed status (default: 100)')
parser.add_argument('--dataset', default='tiny-imagenet-200', 
                    choices=['mnist', 'tiny-imagenet-200'], 
                    help='name of dataset to train on (default: tiny-imagenet-200)')
parser.add_argument('--data-dir', default=os.getcwd(), type=str, 
                    help='path to dataset (default: current directory)')
parser.add_argument('--cuda', action='store_true', 
                    help='cuda (default: True)')
parser.add_argument('--model', default='SVM', choices=['SVM', 'AlexNet'], 
                    help='model to train (default: SVM)')
parser.add_argument('--features', default=12288, type=int, 
                    help='number of input features to SVM (default: 12288)')
parser.add_argument('--classes', default=200, type=int, 
                    help='number of output classes of SVM (default: 200)')
parser.add_argument('--reg', action='store_true', 
                    help='L2 regularization for hinge loss (default: False)')
parser.add_argument('--margin', default=20, type=int, 
                    help='margin for computing hinge loss (default: 20)')
parser.add_argument('--topk', default=1, type=int, 
                    help='top-k accuracy (default: 1)')
parser.add_argument('--results-dir', default=os.path.join(os.getcwd(), 'results'), type=str, 
                    help='path to results (default: current directory)')
parser.add_argument('--prefix', default='default', type=str, 
                    help='prefix of the plot (default: default)')


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
        create_val_img_folder(args)
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
    
    # Plot graphs
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
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    filename = '_'.join([args.prefix, args.dataset, args.model, 'plot.png'])
    fig.savefig(os.path.join(args.results_dir, filename))


if __name__ == '__main__':
    args = parser.parse_args()
    run_experiment(args)
