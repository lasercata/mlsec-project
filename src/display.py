#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Displays results'''

##-Imports
# Python
from os.path import basename
from pathlib import Path

# Project
from src.analyze import main as main_analyze

##-Init
dataType = dict[
    str,
    dict[
        str,
        float |
        list[
            dict[str, int | float | str]
        ]
    ]
]

##-Display
def main_display(json_filenames: list[str], out_dir: str):
    '''
    Entry point for the display function.

    In:
        - json_filenames: the list of json files names
        - out_dir: the output directory to write images
    '''

    json_dct = {
        basename(fn): Path(fn)
        for fn in json_filenames
    }

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    main_analyze(json_dct, out_path)
