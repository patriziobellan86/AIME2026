from PoE3.PoE.ModelRequests import SendToLLM, clean_list, extract_list_items, extract_name, extract_description
from PoE3.PoE.FileToolkit import SaveExpertsFieldList, LoadExpertsFieldList, SaveExperts
from PoE3.PoE.prompts.project_manager import CREATION_LIST_EXPERTS_SYSTEM, CREATION_LIST_EXPERTS_USER, \
    CREATION_LIST_EXPERTS_USER_UNCONSTRAINED, CREATION_LIST_EXPERTS_NO_DESCRIPTION_SYSTEM
from PoE3.PoE.prompts.experts import CREATE_EXPERT_IN_A_FIELD_SYSTEM, CREATE_EXPERT_IN_A_FIELD_USER, \
    CREATE_EXPERT_IN_A_FIELD_NO_PERSONALITY_SYSTEM, CREATE_EXPERT_IN_A_FIELD_NO_PERSONALITY_USER
from PoE3.PoE.prompts.final_decision_maker import CREATION_FINAL_DECISOR_USER, CREATION_FINAL_DECISOR_SYSTEM

import json
from tqdm import tqdm

from PoE3.PoE.utilities import is_with_personality, is_without_description


def get_experts_fields(experts_list):
    """
        return the list of fields of the experts
    """
    fields = list()
    for expert in experts_list:
        field = expert['field']
        fields.append(field)
    return fields


def CreateExpertsFields(args_dict):
    #####   Create Experts Fields #####


    if is_without_description(args_dict):
        #  NO DESCRIPTION
        messages = [{"role": "system",
                     "content": CREATION_LIST_EXPERTS_NO_DESCRIPTION_SYSTEM}]
    else:
        #  either No Personality or with description framework
        messages = [{"role": "system",
                 "content": CREATION_LIST_EXPERTS_SYSTEM.format(
                     projectmanager_description=args_dict['project-manager']['description'])}]


    #  Number of agents
    if args_dict['max_experts_number'] > 1:
        messages.append({"role": "user",
                         "content": CREATION_LIST_EXPERTS_USER.format(task=args_dict['task'],
                                                                      context=args_dict['context'],
                                                                      max_experts_number=args_dict[
                                                                          'max_experts_number'])})
    else:
        messages.append({"role": "user",
                         "content": CREATION_LIST_EXPERTS_USER_UNCONSTRAINED.format(task=args_dict['task'],
                                                                                    context=args_dict['context'])})

    outdict = SendToLLM(args_dict=args_dict,
                        messages=messages,
                        model=args_dict['model'],
                        tokenizer=args_dict['tokenizer'],
                        device=args_dict['device'],
                        temperature=args_dict['temperature'],
                        nucleus=args_dict['nucleus'],
                        alternatives=args_dict['alternatives'],
                        max_tokens=512)
    answer = outdict['response_message']

    try:
        tmp_answer = answer.replace("```json\n", "").replace("```", "")
        tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')

        experts_list = json.loads(tmp_answer)

    except:
        experts_list = extract_list_items(clean_list(args_dict,
                                                     answer,
                                                     model=args_dict['model'],
                                                     tokenizer=args_dict['tokenizer'],
                                                     device=args_dict['device'],
                                                     temperature=0.0,  # temperature,
                                                     nucleus=0.0,  # nucleus,
                                                     alternatives=1,
                                                     max_tokens=512))

    if len(experts_list) > args_dict['max_experts_number'] > 0:
        print(f"\n||| \t\t Experts list too long, selecting the first {args_dict['max_experts_number']}")
        experts_list = experts_list[:args_dict['max_experts_number']]

    print(f'\tExperts list:\t||{experts_list}||')
    outdict['list'] = experts_list
    return outdict


def CreateExpertAgents(args_dict):
    # Load experts list
    experts_list = LoadExpertsFieldList(args_dict)

    #  create experts
    experts = list()
    for expert_id, expert_field in tqdm(enumerate(experts_list), desc="Creating Experts", ascii=True):

        if is_with_personality(args_dict):
            messages = [{"role": "system",
                         "content": CREATE_EXPERT_IN_A_FIELD_SYSTEM.format(
                             psychologist_description=args_dict['psychologist']['description'],
                             description_framework=args_dict['description_framework'])},
                        {"role": "user",
                         "content": CREATE_EXPERT_IN_A_FIELD_USER.format(
                             field=expert_field,
                             description_framework=args_dict['description_framework'])}]
        else:
            if is_without_description(args_dict):
                messages = [{"role": "system",
                             "content": CREATE_EXPERT_IN_A_FIELD_NO_DESCRIPTION_SYSTEM},
                            {"role": "user",
                             "content": CREATE_EXPERT_IN_A_FIELD_NO_DESCRIPTION_USER.format(
                                 field=expert_field)}]
            else:
                # NO-PERSONALITY

                messages = [{"role": "system",
                             "content": CREATE_EXPERT_IN_A_FIELD_NO_PERSONALITY_SYSTEM.format(
                                 psychologist_description=args_dict['psychologist']['description'])},
                            {"role": "user",
                             "content": CREATE_EXPERT_IN_A_FIELD_NO_PERSONALITY_USER.format(
                                 field=expert_field)}]

        outdict = SendToLLM(args_dict=args_dict,
                            messages=messages,
                            model=args_dict['model'],
                            tokenizer=args_dict['tokenizer'],
                            device=args_dict['device'],
                            temperature=args_dict['temperature'],
                            nucleus=args_dict['nucleus'],
                            max_tokens=1024)
        answer = outdict['response_message']
        try:
            tmp_answer = answer.replace("```json\n", "").replace("```", "")
            tmp_dict = json.loads(tmp_answer)
            name = tmp_dict['name']
            description = tmp_dict['description']

        except:
            #  extract data using LLM
            name = extract_name(args_dict=args_dict,
                                list_string=answer,
                                model=args_dict['model'],
                                tokenizer=args_dict['tokenizer'],
                                device=args_dict['device'],
                                temperature=0,
                                nucleus=0,
                                max_tokens=12)
            description = extract_description(args_dict=args_dict,
                                              list_string=answer,
                                              model=args_dict['model'],
                                              tokenizer=args_dict['tokenizer'],
                                              device=args_dict['device'],
                                              temperature=0,
                                              nucleus=0,
                                              max_tokens=1024)
        # print(
        #     f"-------------- Expert Created: {name} field:{expert_field}\nDescription: {description}\n--------------")
        outdict['name'] = name
        outdict['description'] = description
        outdict['field'] = expert_field
        outdict['expert-id'] = expert_id
        experts.append(outdict)

    return experts


def CreateExperts(args_dict):
    CreateExpertsFields(args_dict)
    return CreateExpertAgents(args_dict)
