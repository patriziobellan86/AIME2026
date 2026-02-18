import json

from ExperimentsVirtualPsy.Lunchers.luncher import is_baseline_config
from PoE3.PoE.create_psycologist_agent import create_psychologist_agent_from_config_file
from PoE3.PoE.create_projectmanager import create_project_manager_agent_from_config_file
from PoE3.PoE.create_expertize_fields import create_expertize_fields_from_config_file


from PoE3.PoE.create_expert_agents import create_expert_agents_from_config_file
from PoE3.PoE.create_finaldecisionmake_agent import create_final_decision_maker_agent_from_config_file

from PoE3.PoE.main import InitializeFilesAndFolders
from PoE3.PoE.utilities import check_json_integrity, is_with_personality, is_without_description, is_baseline


def initialize_poe_from_config_file(config_file):
    check_json_integrity(config_file)
    InitializeFilesAndFolders(config_file)

    #  load args_dict file
    with open(config_file, 'r') as json_file:
        args_dict = json.load(json_file)
    if is_baseline(args_dict):
        return

    if is_with_personality(args_dict):
        create_psychologist_agent_from_config_file(config_file)
        create_project_manager_agent_from_config_file(config_file)
        create_expertize_fields_from_config_file(config_file)
        create_expert_agents_from_config_file(config_file)
        create_final_decision_maker_agent_from_config_file(config_file)

    #  NO- PERSONALITY
    else:
        print("Personality not enabled, creating agents without personality traits.")
        if is_without_description(args_dict):
            #  NO-DESCRIPTION
            #  neither psychologist nor project manager agents are created.
            #  PM is created at runtime only to select expertize fileds of EA
            create_expertize_fields_from_config_file(config_file)
            #  NO EAs are created. Instead, they are executed at run-time.
            #  No FDM agent is created. Instead, it is executed at run-time
        else:
            #  NO-PERSONALITY
            create_psychologist_agent_from_config_file(config_file)
            create_project_manager_agent_from_config_file(config_file)
            create_expertize_fields_from_config_file(config_file)
            create_expert_agents_from_config_file(config_file)
            create_final_decision_maker_agent_from_config_file(config_file)


    print('PoE-3 initialization completed')