import argparse
import json
from pathlib import Path

from PoE3.PoE.FileToolkit import SaveProjectManager, save_args_dict, check_project_manager_exist, LoadProjectManager, \
    LoadPsychologist, get_outputdir
from PoE3.PoE.ProjectManager import CreateProjectManager
from PoE3.PoE.main import InitializeFilesAndFolders
from PoE3.PoE.ModelRequests import LoadTokenizerModel
from PoE3.PoE.utilities import is_openrouter, is_with_personality, is_without_description


def create_project_manager_agent_from_config_file(config_file):
    #  load the config file( that is a json)
    with open(config_file, 'r') as json_file:
        config_args_dict = json.load(json_file)
    args_dict_file = Path(get_outputdir(config_args_dict)).joinpath('args_dict.json').absolute().__str__()
    with open(args_dict_file, 'r') as json_file:
        args_dict = json.load(json_file)

    #  Project Manager, used to select expertise fields'
    if check_project_manager_exist(args_dict):
        #  load it
        args_dict['project-manager'] = LoadProjectManager(args_dict)
        print("Project Manager Agent already exists!")
    else:

        if not is_without_description(args_dict):
            args_dict['psychologist'] = LoadPsychologist(args_dict)

        print('***** Creating Project Manager Agent ******')
        if not is_openrouter(args_dict):
            # InitializeFilesAndFolders(args_dict)
            print("loading model and tokenizer")
            #  load model and tokenizer
            if args_dict['tokenizer'] != None:
                # if it is not loaded yet
                print("loading model and tokenizer")
                args_dict['model'], args_dict['tokenizer'], args_dict['device'] = LoadTokenizerModel(args_dict)

        else:
            args_dict['tokenizer'] = ''
            args_dict['device'] = ''

        projectmanager = CreateProjectManager(args_dict)

        # for k, v in projectmanager.items():
        #     print(f"field: {k} value type: {type(v)}")

        SaveProjectManager(args_dict, projectmanager )
        # save_args_dict(args_dict)
        print('***** Project Manager Agent Created ******')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
    parser.add_argument('--config-file',
                        type=str,
                        help='Path to config file',
                        required=True)

    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    create_project_manager_agent_from_config_file(config_file)