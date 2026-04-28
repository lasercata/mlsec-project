#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Implements attacks on models'''

##-Imports
# Python modules
from autoattack import AutoAttack
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.module import Module
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.metrics.classification import Accuracy

# Project
from src.config import Conf

##-Normal accuracy
def calc_normal_accuracy(net: Module, data_loader: DataLoader) -> float:
    '''
    TODO

    In:
        - net
        - data_loader
    '''

    model = BasePyTorchClassifier(net)
    accuracy = Accuracy()(model, data_loader)

    return float(accuracy)

def auto_attack(model: Module, test_loader: DataLoader, eps: float = 8/255, attacks: int = 1) -> float:
    '''
    TODO

    In:
        - model
        - eps: epsilon
        - attacks: if 0, run all, if 1, run just 'apgd-ce'.
    '''

    c = Conf.get_instance()

    if attacks == 1:
        attacks_to_run = ['apgd-ce']
    else:
        attacks_to_run = []

    adversary = AutoAttack(
        model,
        norm=c.vars['THREAT_MODEL'],
        eps=eps,
        version='custom',
        attacks_to_run=attacks_to_run
    )

    adversary.device = c._device
    adversary.apgd.device = c._device
    adversary.apgd.n_restarts = 1

    # Load all data into tensors
    x_test = torch.cat([x for (x, _) in test_loader][:100], 0) #TODO: 100 should be a var
    y_test = torch.cat([y for (_, y) in test_loader][:100], 0)

    x_test.to(c._device)
    y_test.to(c._device)

    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=50) #TODO: 50 should be customizable

    # Get accuracy
    model.eval()
    with torch.no_grad():
        output = model(x_adv)
        pred = output.max(1)[1]
        correct = pred.eq(y_test).float().sum()
        adv_accuracy = correct / y_test.shape[0]

    return float(adv_accuracy)
