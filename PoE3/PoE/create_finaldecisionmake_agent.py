import argparse
import json
from pathlib import Path

from PoE3.PoE.FileToolkit import SaveFinalDecisionMaker, save_args_dict, check_final_decision_maker_exist, \
    check_experts_exist, LoadExperts, LoadPsychologist
from PoE3.PoE.FinalDecisionMaker import CreateFinalDecisionMaker
from PoE3.PoE.ModelRequests import LoadTokenizerModel

from PoE3.PoE.FileToolkit import get_outputdir
from PoE3.PoE.utilities import is_openrouter


def create_final_decision_maker_agent_from_config_file(config_file):
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

    #  create final decisor if it does not exist
    if check_final_decision_maker_exist(args_dict):
        print('\n\t\tFinal decision maker already exist')
        # final_decision_maker = LoadFinalDecisionMaker(args_dict)
    else:
        print('*** Creating Final Decision Maker Agent')
        #  create experts if they do not exist
        if check_experts_exist(args_dict):
            #  load it
            experts = LoadExperts(args_dict)
        else:
            raise FileNotFoundError("Expert Agents do not Exist! Please create them first")

        if 'psychologist' not in args_dict:
            # load pychologist
            args_dict['psychologist'] = LoadPsychologist(args_dict)

        final_decision_maker = CreateFinalDecisionMaker(args_dict, experts)
        print('*** Final Decision Maker Agent Created')

        SaveFinalDecisionMaker(args_dict, final_decision_maker)
        # SaveFinalDecisionMaker(args_dict, final_decision_maker)
        # save_args_dict(args_dict)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
    parser.add_argument('--config-file',
                        type=str,
                        help='Path to config file',
                        required=True)

    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    create_final_decision_maker_agent_from_config_file(config_file)