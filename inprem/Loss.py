#  -*- coding: utf-8 -*-
# @Time : 2019/11/16 下午11:37
# @Author : Xianli Zhang
# @Email : xlbryant@stu.xjtu.edu.cn

import torch
from torch import nn
import torch.autograd.variable as Variable
import numpy as np
import torch.nn.functional as F
import torch.distributions as distributions


class CrossEntropy(nn.Module):
    def __init__(self, task):
        super(CrossEntropy, self).__init__()
        self.task = task
        self.crossEntropy = nn.CrossEntropyLoss()
        self.BCE = nn.BCELoss()

    def forward(self, output, label, train=True):
        loss = None
        if self.task in ['heart', 'diabetes', 'kidney']:
            loss = self.crossEntropy(output, label.long())
        if self.task == 'diagnoses':
            loss = self.BCE(output, label.float())
        return loss


class UncertaintyLoss(nn.Module):
    def __init__(self, task, T, C):
        super(UncertaintyLoss, self).__init__()
        self.task = task
        self.T = T
        self.C = C

    def forward(self, output, label):
        std = torch.sqrt(output[:, self.C:])
        pred = output[:, :self.C]
        dist = distributions.Normal(loc=torch.zeros_like(std), scale=std)
        one_hot_label = torch.zeros(std.shape[0], self.C).scatter_(1, label.unsqueeze(1).cpu(), 1)
        if std.is_cuda:
            one_hot_label = one_hot_label.cuda()
        loss = None
        for t in range(self.T):
            std_sample = torch.transpose(dist.sample(torch.Size([self.C])), 0, 1).squeeze(2)
            distorted_logit = pred + std_sample
            true_logit = torch.sum(distorted_logit * one_hot_label, 1)

            if t == 0:
                loss = torch.exp(true_logit - torch.log(torch.sum(torch.exp(distorted_logit), dim=1)))
            else:
                loss = loss + torch.exp(true_logit - torch.log(torch.sum(torch.exp(distorted_logit), dim=1)))

        return torch.mean(-torch.log(loss / self.T))


