from PoE_smallv2.Experiment import QueriesToExpertAgents
from PoE_smallv2.main import InitializeFilesAndFolders
from PoE_smallv2.ModelRequests import LoadTokenizerModel

import argparse
import json


parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
parser.add_argument('--config-file',
                    type=str,
                    help='Path to config file',
                    required=True)
parser.add_argument('--agent-number',
                    type=int,
                    help='Agent Id (index starts at 0)',
                    required=True)
# Parse the arguments
args = parser.parse_args()
config_file = args.config_file
#  load the config file( that is a json)
with open(config_file, 'r') as json_file:
    args_dict = json.load(json_file)

InitializeFilesAndFolders(args_dict)

print("loading model and tokenizer")
#  load model and tokenizer
args_dict['model'], args_dict['tokenizer'], args_dict['device'] = LoadTokenizerModel(args_dict)
print('\n\n\n\t\t\t\t***** Queries To Expert Agents ******\n\n\n')
QueriesToExpertAgents(args_dict)
print('\n\n\n\t\t\t\t***** Experiment Completed ******\n\n\n')
