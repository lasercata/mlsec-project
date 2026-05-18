#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Implements attacks on models'''

##-Imports
# Python modules
from autoattack import AutoAttack
from datetime import datetime as dt
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.module import Module
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.metrics.classification import Accuracy

# Project
from src.config import Conf, get_nets, get_dataloader

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
        - test_size: the number of samples to take from the `test_loader` (the `test_size` firsts are taken)
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

##-Run all
def run_all(
    calc_acc: bool = True,
    batch_size: int = 1,
    aa_nb_samples: int = 100,
    aa_batch_size: int = 50,
    aa_verbose: bool = False,
    attack_mode: int = 1
) -> dict:
    '''
    Run attacks on all models defined in `config.env` with all epsilon values defined in the same file.

    In:
        - calc_acc: if `True`, calculates the clean accuracy
        - batch_size: the test data batch size
        - aa_nb_samples: the number of samples to run autoattacks on (samples = aa_nb_samples // batch_size)
        - aa_batch_size: the batch size used in autoattack
        - aa_verbose: if True, shows output from auto attack
        - attack_mode: see the `attack` param of function `auto_attack`

    Out:
        A dictionary of the results, in the following shape:
            ```
            {
                "model_name": {
                    "clean_acc": float,
                    "attacks": [
                        {
                            "eps": float,
                            "eps_h": str,
                            "mode": int,
                            "batch_size": int,
                            "aa_nb_samples": int,
                            "aa_batch_size": int,
                            "total_samples": int,
                            "adv_acc": float,
                            "time": float
                        },
                        ...
                    ]
                },
                ...
            }
            ```

    Raises:
        ValueError if aa_nb_samples < batch_size
    '''

    if aa_nb_samples // batch_size == 0:
        raise ValueError('aa_nb_samples < batch_size')

    c = Conf.get_instance()

    t0 = dt.now()
    print('Getting nets...')
    nets = get_nets()
    print(f'Done (getting nets in {dt.now() - t0}s)')

    t1 = dt.now()
    print('Getting data loader...')
    data_loader = get_dataloader(train=False, shuffle=False, batch_size=batch_size)
    print(f'Done (getting data loader in {dt.now() - t1}s)')

    res = {}

    t2 = dt.now()
    print(f'\nStarting attacks at {t2}')
    for m in nets:
        t00 = dt.now()
        print(f'\tModel: {m}, started at {t00}')
        res[m] = {}

        if calc_acc:
            print(f'\tCalculating clean accuracy...')
            acc = calc_normal_accuracy(nets[m], data_loader)
            print(f'\tClean accuracy: {round(100 * acc, 2)}%')
            print(f'\tCalculated in {dt.now() - t00}')

            res[m]['clean_acc'] = acc

        res[m]['attacks'] = []

        for eps in c.vars['EPSILON_VALUES']:
            t_eps_0 = dt.now()
            print(f'\n\t\tAttack on {m} with eps={255*eps}/255 started at {t_eps_0}')

            adv_acc = auto_attack(
                nets[m],
                data_loader,
                eps=eps,
                test_size=(aa_nb_samples // batch_size),
                batch_size=aa_batch_size,
                attacks=attack_mode,
                verbose=aa_verbose
            )

            print(f'\t\tAutoAttack accuracy: <= {round(100 * adv_acc, 2)}%\n')
            print(f'\t\tAttack ran in {dt.now() - t_eps_0}s')

            res[m]['attacks'].append({
                'eps': eps,
                'eps_h': f'{255 * eps}/255',
                'mode': attack_mode,
                'batch_size': batch_size,
                'aa_nb_samples': aa_nb_samples // batch_size,
                'aa_batch_size': aa_batch_size,
                'total_samples': aa_nb_samples,
                'adv_acc': adv_acc,
                'time': (dt.now() - t_eps_0).total_seconds()
            })
    
    print(f'\nAttacks ran in {dt.now() - t2}s')
    print(f'Total elapsed time: {dt.now() - t0}s')

    return res
