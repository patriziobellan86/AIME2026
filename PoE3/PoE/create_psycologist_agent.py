import argparse
import json
from pathlib import Path

from PoE3.PoE.FileToolkit import SavePsychologist, save_args_dict, LoadPsychologist, check_psychologist_exist, \
    get_outputdir
from PoE3.PoE.Psychologist import CreatePsychologist
from PoE3.PoE.utilities import is_openrouter
from PoE3.PoE.ModelRequests import LoadTokenizerModel

def create_psychologist_agent_from_config_file(config_file):
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
        args_dict['device'] =''


    if check_psychologist_exist(args_dict):
        print('Psychologist Agent already exists.')
        args_dict['psychologist'] = LoadPsychologist(args_dict)
    else:
        print('Creating Psychologist Agent')
        args_dict['psychologist'] = CreatePsychologist(args_dict)
        print('Psychologist Agent Created')
        # for k,v in psychologist.items():
        #     print(f"field: {k} value type: {type(v)}")
        #
        SavePsychologist(args_dict,
                         args_dict['psychologist'])

    # save_args_dict(args_dict)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
    parser.add_argument('--config-file',
                        type=str,
                        help='Path to config file',
                        required=True)

    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    create_psychologist_agent_from_config_file(config_file)
