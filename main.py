#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Main file that runs the project'''

##-Imports
# Python modules
from datetime import datetime as dt

# Project
from src.attack import calc_normal_accuracy, auto_attack
from src.config import Conf, get_nets, get_dataloader

##-Run all
def run_all(calc_acc: bool = True, batch_size: int = 1, aa_verbose: bool = False):
    '''
    Run attacks on all models defined in `config.env` with all epsilon values defined in the same file.

    In:
        - calc_acc: if `True`, calculates the clean accuracy
        - batch_size: the test data batch size
        - aa_verbose: if True, shows output from auto attack
    '''

    c = Conf.get_instance()

    t0 = dt.now()
    print('Getting nets...')
    nets = get_nets()
    print(f'Done (getting nets in {dt.now() - t0}s)')

    t1 = dt.now()
    print('Getting data loader...')
    data_loader = get_dataloader(train=False, shuffle=False, batch_size=batch_size)
    print(f'Done (getting data loader in {dt.now() - t1}s)')

    t2 = dt.now()
    print(f'\nStarting attacks at {t2}')
    for m in nets:
        t00 = dt.now()
        print(f'\tModel: {m}, started at {t00}')

        if calc_acc:
            print(f'\tCalculating clean accuracy...')
            acc = calc_normal_accuracy(nets[m], data_loader)
            print(f'\tClean accuracy: {round(100 * acc, 2)}%')
            print(f'\tCalculated in {dt.now() - t00}')

        for eps in c.vars['EPSILON_VALUES']:
            t_eps_0 = dt.now()
            print(f'\n\t\tAttack on {m} with eps={255*eps}/255 started at {t_eps_0}')

            adv_acc = auto_attack(nets[m], data_loader, eps=eps, attacks=1, verbose=aa_verbose)
            print(f'\t\tAutoAttack accuracy: <= {round(100 * adv_acc, 2)}%\n')
            print(f'\t\tAttack ran in {dt.now() - t_eps_0}s')
    
    print(f'\nAttacks ran in {dt.now() - t2}s')
    print(f'Total elapsed time: {dt.now() - t0}s')

##-Main
def main():
    '''Entry point'''

    c = Conf.get_instance()

    print(c.vars)
    print()

    try:
        run_all(calc_acc=False, batch_size=1, aa_verbose=False)
    except KeyboardInterrupt:
        print('Stopped - KeyboardInterrupt')
        return

##-Run
if __name__ == '__main__':
    main()
