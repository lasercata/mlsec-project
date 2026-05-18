#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Main file that runs the project'''

##-Imports
# Project
from src.config import Conf
from src.attack import run_all
from src.parser_ui import ParserUi

##-Main
def main():
    '''Entry point'''

    c = Conf.get_instance()

    print(c.vars)
    print()

    try:
        out = run_all(calc_acc=False, batch_size=1, aa_verbose=False)
        print(out)
    except KeyboardInterrupt:
        print('Stopped - KeyboardInterrupt')
        return

##-Run
if __name__ == '__main__':
    # main()
    app = ParserUi()
    app.parse()
