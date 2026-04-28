#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Reads the config from config.env and exposes constants through a Singleton class'''

##-Imports
import os
from dotenv import load_dotenv
import json

from robustbench.utils import load_model
from torch.nn.modules.module import Module
import torchvision
from torch.utils.data import DataLoader

##-Init
ENV_FN = 'config.env'

##-Config
class Conf:
    '''Singleton class to access config variables'''

    _instance = None

    def __init__(self):
        '''
        Blocks direct initialization.

        Raises:
            RuntimeError
        '''

        raise RuntimeError('Conf: use Conf.get_instance to get the Singleton instance')

    def _init(self):
        '''Initiates the class'''

        # Load config.env
        self._bring_env(ENV_FN)

        # Create attributes from env vars
        self._device = os.environ.get('DEVICE', 'cpu')
        self._datasets_path = os.environ.get('DATASETS_PATH', 'datasets/')
        self._models_path = os.environ.get('MODELS_PATH', 'models/')
        self._dataset = os.environ.get('DATASET', 'cifar10')
        self._threat_model = os.environ.get('THREAT_MODEL', 'Linf')
        self._model_names = json.loads(os.environ.get('MODEL_NAMES', '[]'))


        # Public dict
        self.vars = {
            'DEVICE': self._device,
            'DATASETS_PATH': self._datasets_path,
            'MODELS_PATH': self._models_path,
            'DATASET': self._dataset,
            'THREAT_MODEL': self._threat_model,
            'MODEL_NAMES': self._model_names
        }

    def _bring_env(self, env_fn: str):
        '''Loads the config.env file for "os.environ".'''

        potential_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), env_fn),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env_fn)
        ]

        # Load the first existing file
        for path in potential_paths:
            if os.path.exists(path):
                load_dotenv(path)
                print(f'Loaded environment variables from {path}')
                break

    @staticmethod
    def get_instance() -> Conf:
        '''Returns the Singleton instance, and create it for the first call.'''

        if Conf._instance is None:
            Conf._instance = Conf.__new__(Conf)
            Conf._instance._init()

        return Conf._instance

##-Download
def get_nets() -> dict[str, Module]:
    '''Loads the models and download them if necessary.'''

    c = Conf.get_instance()

    nets = {
        m:
        load_model(
            model_name=m,
            model_dir=c.vars['MODELS_PATH'],
            dataset=c.vars['DATASET'],
            threat_model=c.vars['THREAT_MODEL'],
        )
        for m in c.vars['MODEL_NAMES']
    }

    for net_name in nets:
        nets[net_name].to(c.vars['DEVICE'])

    return nets

def get_dataloader(train: bool, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    '''
    Creates a dataloader.
    It downloads the dataset if not present.

    In:
        - train: if True, creates dataset from training set, otherwise from test set
        - batch_size: how many samples per batch to load
        - shuffle: if True, randomize data order at every epoch

    Out:
        The data loader

    Raises:
        RuntimeError if the environment variable `DATASET` is not valid here (not implemented)
    '''

    c = Conf.get_instance()

    if c.vars['DATASET'].lower() == 'cifar10':
        dataset_loader = torchvision.datasets.CIFAR10
    elif c.vars['DATASET'].lower() == 'cifar100':
        dataset_loader = torchvision.datasets.CIFAR100
    elif c.vars['DATASET'].lower() == 'mnist':
        dataset_loader = torchvision.datasets.MNIST
    else:
        raise RuntimeError(f'environment variable `DATASET` is set to "{c.vars["DATASET"]}", but it is not implemented (only cifar 10, 100 and MNIST are available)')

    dataset = dataset_loader(
        transform=torchvision.transforms.ToTensor(),
        train=train,
        root=c.vars['DATASETS_PATH'],
        download=True
    )

    return DataLoader(dataset, batch_size, shuffle)
