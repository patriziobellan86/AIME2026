import argparse
import json
from pathlib import Path

from PoE3.PoE.CreateExperts import CreateExpertsFields
from PoE3.PoE.FileToolkit import LoadProjectManager, check_project_manager_exist, LoadPsychologist, \
    SaveExpertsFieldList, check_experts_list_exist, get_outputdir
from PoE3.PoE.main import InitializeFilesAndFolders
from PoE3.PoE.ModelRequests import LoadTokenizerModel
from PoE3.PoE.utilities import is_openrouter, is_without_description


def create_expertize_fields_from_config_file(config_file):
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


    if check_experts_list_exist(args_dict):
        print('Expert Agents Expertize List already exist')
    else:
        if is_without_description(args_dict):
            #  NO-DESCRIPTION enabled. agents operated without any description
            pass
        else:
            #  no-personality or using a description framework
            #  Project Manager, used to select expertise fields'
            if check_project_manager_exist(args_dict):
                #  load it
                args_dict['project-manager'] = LoadProjectManager(args_dict)
                if 'psychologist' not in args_dict:
                    # load pychologist
                    args_dict['psychologist'] = LoadPsychologist(args_dict)
            else:
                raise FileNotFoundError("The Project Manager Agent does not exists!")

        #  create expertize fields
        print('***** Creating Expert Agents Expertize List ******')
        outdict = CreateExpertsFields(args_dict)
        print('***** Expertize List Created ******')
        SaveExpertsFieldList(args_dict, outdict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
    parser.add_argument('--config-file',
                        type=str,
                        help='Path to config file',
                        required=True)

    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    create_expertize_fields_from_config_file(config_file)