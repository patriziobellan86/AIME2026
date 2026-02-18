import json

from PoE3.PoE.ModelRequests import SendToLLM, extract_name, extract_description

from PoE3.PoE.prompts.project_manager import CREATION_PROJECTMANAGER_USER, CREATION_PROJECTMANAGER_SYSTEM, \
    CREATION_PROJECTMANAGER_NO_PERSONALITY_USER, CREATION_PROJECTMANAGER_NO_PERSONALITY_SYSTEM
from PoE3.PoE.utilities import is_with_personality, is_without_description


def CreateProjectManager(args_dict):
    if is_with_personality(args_dict):
        messages = [{"role": "system",
                     "content": CREATION_PROJECTMANAGER_SYSTEM.format(
                         psychologist_description=args_dict['psychologist']['description'],
                         description_framework=args_dict['description_framework'])},
                    {"role": "user",
                     "content": CREATION_PROJECTMANAGER_USER.format(task=args_dict['task'],
                                                                    context=args_dict['context'],
                                                                    description_framework=args_dict[
                                                                        'description_framework'])}]
    else:
        # No Personality
        messages = [{"role": "system",
                     "content": CREATION_PROJECTMANAGER_NO_PERSONALITY_SYSTEM.format(
                         psychologist_description=args_dict['psychologist']['description'])},
                    {"role": "user",
                     "content": CREATION_PROJECTMANAGER_NO_PERSONALITY_USER.format(task=args_dict['task'],
                                                                    context=args_dict['context'])}]
    outdict = SendToLLM(args_dict=args_dict,
                        messages=messages,
                        model=args_dict['model'],
                        tokenizer=args_dict['tokenizer'],
                        device=args_dict['device'],
                        temperature=args_dict['temperature'],
                        nucleus=args_dict['nucleus'],
                        alternatives=args_dict['alternatives'],
                        max_tokens=1024)
    answer = outdict["response_message"]
    #  if the data is parsable as JSON use it, otherwise, use the extracted data
    try:
        tmp_answer = answer.replace("```json\n", "").replace("```", "")
        tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')
        tmp_dict = json.loads(tmp_answer)
        name = tmp_dict['name']
        description = tmp_dict['description']

    except:
        name = extract_name(args_dict,
                            list_string=answer,
                            model=args_dict['model'],
                            tokenizer=args_dict['tokenizer'],
                            device=args_dict['device'],
                            temperature=0,
                            nucleus=0,
                            max_tokens=12)
        description = extract_description(args_dict,
                            list_string=answer,
                                          model=args_dict['model'],
                                          tokenizer=args_dict['tokenizer'],
                                          device=args_dict['device'],
                                          temperature=0,
                                          nucleus=0,
                                          max_tokens=1024)
    print(f"Created Project Manager: \t{name}\nDescription: \t{description}")
    #  save the data generated up to here
    outdict["name"] = name
    outdict["description"] = description
    return outdict
