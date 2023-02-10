import numpy as np
import os
import torch.nn.functional as F
import torch.autograd
import pdb

class Acc(object):
    def __init__(self):
        self.sum_metric = 0
        self.num_inst = 0

    def update(self, predict, label, loss):
        assert predict.dim() == label.dim()

        predict = predict.data
        label = label.data
        if_match = (predict == label)
        self.sum_metric += if_match.float().sum()
        self.num_inst += np.prod(if_match.shape).astype(float)

    def get(self):
        if self.num_inst == 0:
            return float('nan')
        else:
            return (self.sum_metric / self.num_inst)

    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0


class TrainingLoss(object):
    def __init__(self):
        self.sum_metric = 0
        self.num_inst = 0

    def update(self, predict, label, loss):
        self.sum_metric += loss.sum()
        self.num_inst += np.prod(loss.shape).astype(float)

    def get(self):
        if self.num_inst == 0:
            return float('nan')
        else:
            return (self.sum_metric / self.num_inst)

    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0
