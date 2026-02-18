from PoE3.PoE.ModelRequests import SendToLLM, extract_description, extract_name
from PoE3.PoE.prompts.psychologist import CREATION_PSYCHOLOGIST_SYSTEM, CREATION_PSYCHOLOGIST_USER, \
    CREATION_PSYCHOLOGIST_constrained_SYSTEM, CREATION_PSYCHOLOGIST_constrained_USER, \
    CREATION_PSYCHOLOGIST_NO_PERSONALITY_SYSTEM, CREATION_PSYCHOLOGIST_NO_PERSONALITY_USER
import json

from PoE3.PoE.utilities import is_with_personality


def CreatePsychologist(args_dict):

    if 'constrain-psychologist' in args_dict:
        messages = [{"role": "system",
                     "content": CREATION_PSYCHOLOGIST_constrained_SYSTEM.format(
                         psychologist_constraint=args_dict['constrain-psychologist'],
                         description_framework=args_dict['description_framework'])
                     },
                    {"role": "user",
                     "content": CREATION_PSYCHOLOGIST_constrained_USER.format(
                         psychologist_constraint=args_dict['constrain-psychologist'],
                         description_framework=args_dict['description_framework'])}]

    elif not is_with_personality(args_dict):
        messages = [{"role": "system",
                         "content": CREATION_PSYCHOLOGIST_NO_PERSONALITY_SYSTEM
                         },
                        {"role": "user",
                         "content": CREATION_PSYCHOLOGIST_NO_PERSONALITY_USER}]
    else:

        messages = [{"role": "system",
                     "content": CREATION_PSYCHOLOGIST_SYSTEM.format(
                         description_framework=args_dict['description_framework'])
                     },
                    {"role": "user",
                     "content": CREATION_PSYCHOLOGIST_USER.format(
                         description_framework=args_dict['description_framework'])}]
    outdict = SendToLLM(args_dict=args_dict,
                            messages=messages,
                            model=args_dict['model'],
                            tokenizer=args_dict['tokenizer'],
                            device=args_dict['device'],
                            temperature=args_dict['temperature'],
                            nucleus=args_dict['nucleus'],
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
                            temperature=args_dict['temperature'],
                            nucleus=args_dict['nucleus'],
                            max_tokens=12)
        description = extract_description(args_dict,
                                          list_string=answer,
                                          model=args_dict['model'],
                                          tokenizer=args_dict['tokenizer'],
                                          device=args_dict['device'],
                                          temperature=args_dict['temperature'],
                                          nucleus=args_dict['nucleus'],
                                          max_tokens=1024)
    print(f"--------------\nCreated Psychologist: {name}\nDescription: {description}\n--------------")
    #  save the data generated up to here
    outdict["name"] = name
    outdict["description"] = description
    return outdict
