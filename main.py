#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Main file that runs the project'''

##-Imports
# Python modules

# Project
from src.attack import calc_normal_accuracy, auto_attack
from src.config import Conf, get_nets, get_dataloader

##-Testing TODO
def test_this():
    '''TODO'''

    print('Getting nets...')
    nets = get_nets()
    print('Done (getting nets)')

    print('Getting data loader...')
    data_loader = get_dataloader(train=False, shuffle=False, batch_size=5)
    print('Done (getting data loader)')

    print('\nCalculate accuracy for models...')
    for m in nets:
        print(f'\tFor model "{m}"...')
        acc = calc_normal_accuracy(nets[m], data_loader)
        
        print(f'\tClean accuracy: {round(100 * acc, 2)}%\n')

def test_that():
    '''TODO'''

    print('Getting nets...')
    nets = get_nets()
    print('Done (getting nets)')

    print('Getting data loader...')
    data_loader = get_dataloader(train=False, shuffle=False, batch_size=5)
    print('Done (getting data loader)')

    print('\nCalculate accuracy for models...')
    for m in nets:
        print(f'\tAccuracy for model "{m}"...')
        # acc = calc_normal_accuracy(nets[m], data_loader)
        # print(f'\tClean accuracy: {round(100 * acc, 2)}%')

        adv_acc = auto_attack(nets[m], data_loader)
        print(f'\tAutoAttack accuracy: <= {round(100 * adv_acc, 2)}% (only one attack)\n')

##-Main
def main():
    '''Entry point'''

    c = Conf.get_instance()

    print(c.vars)
    print()

    try:
        test_that()
    except KeyboardInterrupt:
        print('Stopped - KeyboardInterrupt')
        return

##-Run
if __name__ == '__main__':
    main()
