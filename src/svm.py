#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn


class SVM(nn.Module):
    def __init__(self, n_feature, n_class):
        super(SVM, self).__init__()
        self.fc = nn.Linear(n_feature, n_class)

    def forward(self, x):
        x = self.fc(x)
        return x


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
