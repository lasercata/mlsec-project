#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Reads the config from config.env and exposes constants through a Singleton class'''

##-Imports
import os
from dotenv import load_dotenv

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
        self._dataset_path = os.environ.get('DATASET_PATH', 'datasets/')

        # Public dict
        self.vars = {
            'DEVICE': self._device,
            'DATASET_PATH': self._dataset_path
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
