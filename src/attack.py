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
    Calculates the (initial / clean) accuracy of the model `net`.

    In:
        - net
        - data_loader
    '''

    model = BasePyTorchClassifier(net)
    accuracy = Accuracy()(model, data_loader)

    return float(accuracy)

def auto_attack(
    model: Module,
    test_loader: DataLoader,
    eps: float = 8/255,
    test_size: int = 100,
    batch_size: int = 50,
    attacks: int = 1,
    verbose: bool = True
) -> float:
    '''
    Run the attacks on a given model with a given epsilon value.

    In:
        - model
        - test_loader: the loader for the data to be used in the attacks
        - eps: epsilon (default: 8/255)
        - test_size: the number of samples to take from the `test_loader` (the `test_size` first are taken)
        - batch_size: the batch size in auto attack (default: 50)
        - attacks: if 0, run all, if 1, run just 'apgd-ce', if 2, run just 'apgd-ce' and 'apgd-t'
        - verbose: if `True`, display AutoAttack's progress
    '''

    c = Conf.get_instance()

    if attacks == 1:
        attacks_to_run = ['apgd-ce']
        a_version='custom'
    elif attacks == 2:
        attacks_to_run = ['apgd-ce', 'apgd-t']
        a_version='custom'
    else:
        attacks_to_run = []
        a_version='standard'

    adversary = AutoAttack(
        model,
        norm=c.vars['THREAT_MODEL'],
        eps=eps,
        version=a_version,
        attacks_to_run=attacks_to_run,
        device=c.vars['DEVICE'],
        verbose=verbose
    )

    adversary.device = c.vars['DEVICE']
    adversary.apgd.device = c.vars['DEVICE']
    adversary.apgd.n_restarts = 1

    # Load all data into tensors
    x_test = torch.cat([x for (x, _) in test_loader][:test_size], 0)
    y_test = torch.cat([y for (_, y) in test_loader][:test_size], 0)

    x_test.to(c.vars['DEVICE'])
    y_test.to(c.vars['DEVICE'])

    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)

    # Get accuracy
    model.eval()
    with torch.no_grad():
        output = model(x_adv)
        pred = output.max(1)[1]
        correct = pred.eq(y_test).float().sum()
        adv_accuracy = correct / y_test.shape[0]

    return float(adv_accuracy)
