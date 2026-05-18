#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Defines the user interface using argparse.'''

##-Imports
# Python
import argparse
from datetime import datetime as dt
import json
from os.path import isfile

# Project
from src.attack import run_all
from src.display import main_display

##-Init
version = 'v1.0'

##-Utils
def get_file_content(fn: str, parser=None) -> str:
    '''
    Try to read the file `fn`.
    If not found and `parser` != None, raise an error with `parser.error`. If `parser` is None, raise an `ArgumentTypeError`.
    '''

    try:
        with open(fn, 'r') as f:
            content = f.read()

    except FileNotFoundError:
        if parser != None:
            parser.error(f'The file {fn} has not been found')
        else:
            raise argparse.ArgumentTypeError(f'The file {fn} has not been found')

    return content

##-Ui parser
class ParserUi:
    '''Defines an argument parser'''

    def __init__(self):
        '''Initiate Parser'''

        #------Main parser
        #---Init
        examples = 'Examples :'
        examples += '\n\t./main.py c -h'
        examples += '\n\t./main.py c -o out.json'
        examples += '\n\t./main.py s out.json'

        self.parser = argparse.ArgumentParser(
            description='Evaluation of models from robust bench',
            epilog=examples
        )

        #---Add arguments
        self.parser.add_argument(
            '-V', '--version',
            action='version',
            version='%(prog)s ' + version
        )

        #---Sub parsers
        self.subparsers = self.parser.add_subparsers(required=True, dest='subparser')

        self.create_calculate()
        self.create_show()

    def create_calculate(self):
        '''Creates the calculate subparser and add its arguments.'''
    
        #---Init
        self.parser_c = self.subparsers.add_parser('calculate', aliases=['c', 'calc'], help='run the attacks and export the results')

        #---Add arguments
        self.parser_c.add_argument(
            '-o', '--output',
            dest='output',
            action='store',
            default='',
            help='name of the output file (default: stdout)'
        )

        self.parser_c.add_argument(
            '-q', '--quiet',
            action='store_true',
            help='suppress some output from autoattack'
        )
        self.parser_c.add_argument(
            '-n', '--no-accuracy',
            action='store_true',
            help='do not calculate clean accuracy'
        )
        self.parser_c.add_argument(
            '-a', '--attack-mode',
            type=int,
            default=0,
            help='0 for standard attack; 1 for only apgd-ce; 2 for apgd-ce + apgd-t'
        )
        self.parser_c.add_argument(
            '-s', '--sample-size',
            type=int,
            default=100,
            help='number of samples to use in AutoAttack (grouped by --batch-size)'
        )
        self.parser_c.add_argument(
            '-b', '--batch-size',
            type=int,
            default=5,
            help='batch size for test data'
        )
        self.parser_c.add_argument(
            '-B', '--aa-batch-size',
            type=int,
            default=50,
            help='batch size for attacks data'
        )

    def create_show(self):
        '''Creates the show subparser and add its arguments.'''
    
        #---Init
        self.parser_s = self.subparsers.add_parser('show', aliases=['s'], help='Displays results from a json file')

        #---Add arguments
        self.parser_s.add_argument(
            'inputfile',
            type=str,
            nargs=1,
            help='name of the input json file'
        )

        # self.parser_s.add_argument(
        #     '-o', '--output',
        #     dest='outputfile',
        #     action='store',
        #     default='',
        #     help='name of the output file (default: stdout)'
        # )
        pass

    def parse(self):
        '''Parse the arguments of the main parser, to redirect to the right parser.'''

        #---Get arguments
        args = self.parser.parse_args()

        #---Redirect towards the right method
        if args.subparser in ('c', 'calc', 'calculate'):
            self.parse_calculate(args)
        elif args.subparser in ('s', 'show'):
            self.parse_show(args)

    def parse_calculate(self, args):
        '''Parse the arguments for the `analyse` mode'''

        res = run_all(
            calc_acc=(not args.no_accuracy),
            batch_size=args.batch_size,
            aa_batch_size=args.aa_batch_size,
            aa_nb_samples=args.sample_size,
            aa_verbose=(not args.quiet),
            attack_mode=args.attack_mode
        )

        if args.output == '':
            print()
            print(json.dumps(res, indent=2))
        else:
            if isfile(args.output):
                end = f'_{(dt.now()).isoformat()}'

                if args.output[-5:] == '.json':
                    out_fn = args.output[:-5] + end + '.json'
                else:
                    out_fn = args.output + end

            else:
                out_fn = args.output

            with open(out_fn, 'w') as f:
                json.dump(res, f, indent=2)

    def parse_show(self, args):
        '''Parse the arguments for the `compile` mode'''

        data = get_file_content(args.inputfile[0], self.parser_s)

        try:
            stats = json.loads(data)

        except json.JSONDecodeError as err:
            self.parser_s.error(f'cannot decode json input file: {err}')

        main_display(stats)
