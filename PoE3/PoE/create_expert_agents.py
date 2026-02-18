import argparse
import json
from pathlib import Path

from PoE3.PoE.CreateExperts import CreateExpertAgents
from PoE3.PoE.FileToolkit import save_args_dict, \
    check_experts_exist, LoadExperts, SaveExperts, LoadPsychologist, get_outputdir
from PoE3.PoE.ModelRequests import LoadTokenizerModel
from PoE3.PoE.utilities import is_openrouter


def create_expert_agents_from_config_file(config_file):
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

    #  create experts if they do not exist
    if check_experts_exist(args_dict):
        #  load it
        try:
            experts = LoadExperts(args_dict)
            if len(experts)> 0:
                print('Expert Agents already exist')
                return
        except:
            pass

    print('***** Creating Experts Agents ******')
    if 'psychologist' not in args_dict and not ('no-description' in args_dict and  args_dict['no-description']):
        # load pychologist
        args_dict['psychologist'] = LoadPsychologist(args_dict)

    experts = CreateExpertAgents(args_dict)
    SaveExperts(args_dict, experts)
    print('Expert Agents Created')
    # save_args_dict(args_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
    parser.add_argument('--config-file',
                        type=str,
                        help='Path to config file',
                        required=True)

    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file

    create_expert_agents_from_config_file(config_file)