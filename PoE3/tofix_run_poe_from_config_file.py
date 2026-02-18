import argparse
import json
from create_psycologist_agent_from_config_file import create_psychologist_agent_from_config_file
from create_projectmanager_from_config_file import create_project_manager_agent_from_config_file
from create_expert_agents_from_config_file import create_expert_agents_from_config_file
from create_finaldecisionmake_agent_from_config_file import create_final_decision_maker_agent_from_config_file
from run_queries_to_expert_agents import run_queries_to_expert_agents_from_config_file
from run_queries_to_final_decision_maker_agent import run_final_decision_maker_from_config_file
from PoE_smallv2.main import InitializeFilesAndFolders

def initialize_poe_from_config_file(config_file):

    create_psychologist_agent_from_config_file(config_file)
    create_project_manager_agent_from_config_file(config_file)
    create_expert_agents_from_config_file(config_file)
    create_final_decision_maker_agent_from_config_file(config_file)

    run_queries_to_expert_agents_from_config_file(config_file)
    run_final_decision_make_from_config_file(config_file)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
    parser.add_argument('--config-file',
                        type=str,
                        help='Path to config file',
                        required=True)

    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    initialize_poe_from_config_file(config_file)