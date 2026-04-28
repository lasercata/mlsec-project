#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Main file that runs the project'''

##-Imports
# Python modules

# Project
from src.config import Conf

##-Main
def main():
    '''Entry point'''

    c = Conf.get_instance()

    print(c.vars)

    pass

##-Run
if __name__ == '__main__':
    main()
