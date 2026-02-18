from pathlib import Path

from PoE3.PoE.Baseline import RunBaseline
from PoE3.PoE.FileToolkit import get_outputdir
from PoE3.PoE.main import InitializeFilesAndFolders
from PoE3.PoE.ModelRequests import LoadTokenizerModel

import argparse
import json
import datetime

from PoE3.PoE.utilities import is_openrouter


def run_baseline_from_config_file(config_file):
    #  load the config file( that is a json)
    with open(config_file, 'r') as json_file:
        config_args_dict = json.load(json_file)
    args_dict_file = Path(get_outputdir(config_args_dict)).joinpath('args_dict.json').absolute().__str__()
    with open(args_dict_file, 'r') as json_file:
        args_dict = json.load(json_file)

    if not is_openrouter(args_dict):
        # InitializeFilesAndFolders(args_dict)
        print("loading model and tokenizer")
        #  load model and tokenizer
        args_dict['model'], args_dict['tokenizer'], args_dict['device'] = LoadTokenizerModel(args_dict)


    else:
        args_dict['tokenizer'] = ''
        args_dict['device'] = ''

    print('***** Running Baseline ******')
    RunBaseline(args_dict)
    print('***** Experiment Completed ******')

