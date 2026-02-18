import json

from PoE3.PoE.prompts.tools_instructions import CLEAN_LIST_SYSTEM, CLEAN_LIST_USER, EXTRACT_BASE_USER, \
    EXTRACT_NAME_SYSTEM, EXTRACT_DESCRIPTION_SYSTEM, EXTRACT_JUSTIFICATION_SYSTEM, \
    EXTRACT_GRADE_SYSTEM, EXTRACT_FINAL_ANSWER_SYSTEM, \
    EXTRACT_CONFIDENCE_SCORE_SYSTEM, EXTRACT_REASONING_STEPS_SYSTEM, EXTRACT_CONCLUSION_SYSTEM

from PoE3.PoE.prompts.chat_template import chat_template
import requests
from time import time
# import numpy as np
from nltk import sent_tokenize

import os

from PoE3.PoE.utilities import is_openrouter, is_without_description

#  set seed

seed_value = 23

try:
    from transformers import set_seed
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import warnings
    import fpet

    # Set seed for PyTorch
    torch.manual_seed(seed_value)

    # Set seed for Hugging Face Transformers
    set_seed(seed_value)

except ImportError:
    # compatibility for openrouter only

    pass

try:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    #
    # # Suppress specific warning from configuration_utils.py
    # warnings.filterwarnings("ignore", message="`do_sample` is set to `False`. However, `temperature` is set to `0.0`")
    #
except:
    pass

PATTERN_TO_REMOVE = "<|start_header_id|>assistant<|end_header_id|>"


##################################
#  initialize the model
##################################


#################################

# # cache_dir = '/PoE/.cache'  # '/storage/IDA/Patrizio/PoE/cache' #  '/mnt/md0/data/' #
# token = "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"
# token = "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"  #  3.2 1B

# device = 'cuda:0'


def LoadTokenizerModel(args_dict):
    """
        args_dict is a dictionary containing at least:
            'model' (str): name of the model to load
            'token'      (str): API token
            'cache_dir'  (str): cache directory

        return:
            model, tokenizer, and the device to map input
    """

    tokenizer = AutoTokenizer.from_pretrained(args_dict['model'],
                                              token=args_dict['token'],
                                              cache_dir=args_dict['cache_dir'])
    # Set the pad_token to eos_token if pad_token doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set the chat template in the tokenizer
    tokenizer.chat_template = chat_template
    # print('REMOVE CPU FROM MODEL REQUESTS')
    # # Use init_empty_weights to allocate model layers with empty weights
    # model = AutoModelForCausalLM.from_pretrained(
    #     args_dict['model'],
    #     device_map='cpu',
    #     use_auth_token=args_dict['token'],  # Use the correct parameter name
    #     cache_dir=args_dict['cache_dir']
    # )

    # # original
    model = AutoModelForCausalLM.from_pretrained(
        args_dict['model'],
        torch_dtype=torch.float16,  # Use FP16 precision to save memory
        device_map='auto',  # Automatically select device (CPU/GPU)
        token=args_dict['token'],
        cache_dir=args_dict['cache_dir']
    )

    print("Model and tokenizer are ready for inference.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")
    return model, tokenizer, device


##################################
def SendToLLM(args_dict,
              messages, model=None,
              tokenizer=None,
              device=None,
              temperature=1.2,
              nucleus=0.0,
              alternatives=2,
              max_tokens=1024):
    if is_openrouter(args_dict):
        return SendToLLMOpenRouter(messages=messages,
                                   args_dict=args_dict,
                                   temperature=temperature,
                                   nucleus=nucleus,
                                   max_tokens=max_tokens
                                   )
    else:
        return SendToLLMLocal(messages=messages,
                              model=model,
                              tokenizer=tokenizer,
                              device=device,
                              temperature=temperature,
                              nucleus=nucleus,
                              alternatives=alternatives,
                              max_tokens=max_tokens)


def SendToLLMLocal(messages: list,
                   model=None,
                   tokenizer=None,
                   device=None,
                   temperature=1.2,
                   nucleus=0.0,
                   alternatives=2,
                   max_tokens=1024) -> dict:
    """

    :param messages:  the list of messages
    :param model:  the Large language model
    :param temperature:  the temperature
    :param max_tokens: the max number of tokens generated

    :return: response_message, messages, generation_probability.item(), generation_time

            NB: the list of messages is updated even if it is not returned. since it is a list. remember this behavior in python
    """
    # print(f"device: {device}")

    # Tokenize Messages
    inputs = tokenizer.apply_chat_template(conversation=messages,
                                           chat_template=chat_template,
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_dict=True,
                                           padding=True,
                                           return_tensors="pt",
                                           continue_final_message=False).to(device)

    # print(tokenizer.decode(inputs['input_ids']))
    input_ids = inputs['input_ids']  # Extract the input_ids tensor
    if input_ids.shape[1] > int(max_tokens * 0.9):
        # print(f"max_tokens: {max_tokens} | {input_ids.shape[1]}")
        max_tokens = input_ids.shape[1] + max_tokens

    #
    # print(f"input_ids device: {input_ids.device}")
    # print(f"model device: {next(model.parameters()).device}")

    # print(f"max_tokens: {max_tokens}")
    #
    # # Convert each tensor of token IDs back to a list of integers
    # decoded_inputs = [tokenizer.decode(ids, skip_special_tokens=False) for ids in input_ids.tolist()]

    # # Print the decoded inputs
    # for i, decoded_text in enumerate(decoded_inputs):
    #     print(f"Decoded Input {i}: {type(decoded_text)=} {decoded_text=}")
    #

    # Generate response
    if temperature == 0.0 and nucleus == 0.0:
        do_sampling = False
        alternatives = 1
        temperature = None
        nucleus = None
    elif temperature == 0.0 and nucleus != 0.0:
        #  set temperature to 0.1
        temperature = 0.1
        do_sampling = True
    else:
        do_sampling = True

    # print(f"do_sampling: {do_sampling} | {alternatives=} | {temperature=} | {nucleus=}")
    # Ensure max_length does not exceed the model's capacity
    # max_length = min(max_length, model.config.max_position_embeddings)
    start_time = time()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):  # Use mixed precision for better performance
            with fpet.autocast():  # fpet integration for further optimization

                # Generate text
                # model.eval()
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=do_sampling,
                    temperature=temperature,
                    top_p=nucleus,
                    # top_k=1550,
                    num_return_sequences=alternatives,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                transition_scores = model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True  # normalize SHOULD be true?
                )
    generation_time = time() - start_time

    # print(f"Time taken to generate: {time() - start_time:.2f}s")
    #  move from cuda to cpu
    transition_scores = transition_scores.cpu()

    # output_length = inputs.input_ids.shape[1] + np.sum(transition_scores.numpy() < 0, axis=1)
    # print(f"output_length: {output_length}")
    # length_penalty = model.generation_config.length_penalty

    probabilities = torch.exp(transition_scores.sum(axis=1) / 1)  # no penality given (output_length, length_penalty)
    #  convert to float
    probabilities = probabilities.float()
    # print(f"probabilities: {probabilities}")

    generated_sequences = [tokenizer.decode(seq, skip_special_tokens=False) for seq in outputs['sequences']]
    # text_prob = zip(generated_sequences, probabilities.tolist())
    # take the highest probability
    high_prob_text = generated_sequences[probabilities.argmax()]
    generation_probability = probabilities.max()

    #  remove input messages from the generated text
    # print(f"high_prob_text: p({generation_probability}) | {high_prob_text}")

    response_message = high_prob_text[high_prob_text.rfind(PATTERN_TO_REMOVE) + len(PATTERN_TO_REMOVE):]
    response_message = response_message.replace("<|eot_id|>", "").strip()

    generated_sequences_clean = list()
    for generated_sequence in generated_sequences:
        generated_sequence = generated_sequence[generated_sequence.rfind(PATTERN_TO_REMOVE) + len(PATTERN_TO_REMOVE):]
        generated_sequence = generated_sequence.replace("<|eot_id|>", "").strip()
        generated_sequences_clean.append(generated_sequence)

    # print(f"response_message: {response_message}, {generation_probability=}")
    outdict = {
        "response_message": response_message,

        "generation_probability": generation_probability.item(),
        "generation_time": generation_time,
        "messages": messages,

        "generated_sequences": generated_sequences_clean,
        "probabilities": probabilities.tolist(),
    }
    return outdict
    # return response_message, messages, generation_probability.item(), generation_time


def SendToLLMOpenRouter(messages: list,
                        args_dict: dict,
                        temperature=1.2,
                        nucleus=0.0,
                        max_tokens=150
                        ) -> dict:
    """

    :param messages:  the list of messages
    :param args_dict:  the config_file
    :param temperature:  the temperature
    :param max_tokens: the max number of tokens generated

    :return: response_message, messages, generation_probability.item(), generation_time

    """

    #  load openrouter config file
    with open(args_dict['openrouterfile'], 'r') as f:
        openrouter_config = json.load(f)

    apikey = open(openrouter_config["headers"]["Authorization"], 'r').read().strip()
    openrouter_config["headers"]["Authorization"] = f"Bearer {apikey}"
    openrouter_config["dumps"]["messages"] = messages

    start_time = time()
    response = requests.post(
        url=openrouter_config["url"],
        headers=openrouter_config["headers"],
        data=json.dumps(openrouter_config["dumps"])
    )
    generation_time = time() - start_time

    text_response_json = json.loads(response.text)
    try:
        response_message = text_response_json['choices'][0]['message']['content']
    except KeyError: #KeyError: 'choices'
        try:
            if text_response_json['error']['code'] == 403:
                response_message = f"{text_response_json['error']['code']} {text_response_json['error']['message']}"
            else:
                print(f"Error in response: {text_response_json}")
                raise KeyError("The response does not contain 'choices' or 'message' keys.")
        except:
            print(f"Error in response: {text_response_json}")
            raise KeyError("The response does not contain 'choices' or 'message' keys.")

    outdict = {
        "response_message": response_message,

        "generation_probability": 'unknown',
        "generation_time": generation_time,
        # "messages": messages,

        "generated_sequences": [response_message],
        "probabilities": []
    }
    return outdict


def update_messages(messages: list,
                    role: str,
                    query: str,
                    ):
    messages.append({"role": role,
                     "content": query})
    return messages


def clean_list(args_dict,
               list_string: str,
               model=None,
               tokenizer=None,
               device=None,
               temperature=1.2,
               nucleus=0.0,
               alternatives=2,
               max_tokens=4096) -> str:
    """

    :param list_string: (str) the list to clean
    :param model:  (str)the Large language model
    :param temperature: (float) the temperature
    :param max_tokens: (int) the max number of tokens generated
    :return: (str)the text generated

    """

    messages = list()
    messages.append({"role": "system",
                     "content": CLEAN_LIST_SYSTEM})
    messages.append({"role": "user",
                     "content": CLEAN_LIST_USER.format(text=list_string)})
    outdict = SendToLLM(
        args_dict=args_dict,
        messages=messages,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        alternatives=alternatives,
                        max_tokens=max_tokens)
    return outdict["response_message"]
    # return response_message


def extract_list_items(list_string: str) -> list:
    return [item.strip() for item in list_string.split('\n')]


def extract_base(args_dict: dict, base_prompt: str,
                 string: str,
                 model=None,
                 tokenizer=None,
                 device=None,
                 temperature=1.2,
                 nucleus=0.0,
                 max_tokens=4096,
                 ) -> str:
    """

        base_prompt is the prompt to use to extract the data from

                    Base extract method

    """

    messages = list()
    messages.append({"role": "system",
                     "content": base_prompt})
    messages.append({"role": "user",
                     "content": EXTRACT_BASE_USER.format(text=string)})
    outdict = SendToLLM(
        args_dict=args_dict,
        messages=messages,
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=temperature,
        nucleus=nucleus,
        max_tokens=max_tokens)
    return outdict["response_message"]


def extract_name(args_dict: dict,
                 list_string: str,
                 model=None,
                 tokenizer=None,
                 device=None,
                 temperature=1.2,
                 nucleus=0.0,
                 max_tokens=64) -> str:
    return extract_base(args_dict=args_dict,
                        base_prompt=EXTRACT_NAME_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_description(args_dict: dict, list_string: str,
                        model=None,
                        tokenizer=None,
                        device=None,
                        temperature=1.2,
                        nucleus=0.0,
                        max_tokens=4096) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_DESCRIPTION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_grade(args_dict: dict, list_string: str,
                  model=None,
                  tokenizer=None,
                  device=None,
                  temperature=1.2,
                  nucleus=0.0,
                  max_tokens=32) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_GRADE_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_justification(args_dict: dict, list_string: str,
                          model=None,
                          tokenizer=None,
                          device=None,
                          temperature=1.2,
                          nucleus=0.0,
                          max_tokens=248) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_JUSTIFICATION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_final_answer(args_dict: dict, list_string: str,
                         model=None,
                         tokenizer=None,
                         device=None,
                         temperature=1.2,
                         nucleus=0.0,
                         max_tokens=1024) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_FINAL_ANSWER_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_confidence_score(args_dict: dict, list_string: str,
                             model=None,
                             tokenizer=None,
                             device=None,
                             temperature=1.2,
                             nucleus=0.0,
                             max_tokens=16) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_CONFIDENCE_SCORE_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_reasoning_steps(args_dict: dict,
                            list_string: str,
                            model=None,
                            tokenizer=None,
                            device=None,
                            temperature=1.2,
                            nucleus=0.0,
                            max_tokens=2048) -> list:
    outdict = extract_base(args_dict=args_dict, base_prompt=EXTRACT_REASONING_STEPS_SYSTEM,
                           string=list_string,
                           model=model,
                           tokenizer=tokenizer,
                           device=device,
                           temperature=temperature,
                           nucleus=nucleus,
                           max_tokens=max_tokens,
                           )
    return sent_tokenize(outdict, language='english')
    # return extract_list_items(response_message)


def extract_conclusion(args_dict: dict, list_string: str,
                       model=None,
                       tokenizer=None,
                       device=None,
                       temperature=1.2,
                       nucleus=0.0,
                       max_tokens=1024) -> str:
    return extract_base(args_dict=args_dict, base_prompt=EXTRACT_CONCLUSION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def create_experts_answers_string(query_answers: list, experts: list) -> str:

    #  experiments with description
    experts_answers_string = '[\n'
    for ans, expert in zip(query_answers, experts):
        expert_string = "{\n"
        try:
            expert_string += f"\"expert-name\": \"{expert['name']}\",\n"
        except KeyError:
            pass
        expert_string += f"\"expert-field\": \"{expert['field']}\",\n"
        expert_string += f"\"answer\": \"{ans['final_answer']}\",\n"
        expert_string += f"\"grade\": \"{ans['grade']}\",\n"
        expert_string += f"\"confidence-score\": \"{ans['confidence_score']}\",\n"
        expert_string += f"\"justification\": \"{ans['justification']}\",\n"
        expert_string += f"\"reasoning-steps\": \"{ans['reasoning_steps']}\",\n"
        expert_string += f"\"conclusion\": \"{ans['conclusion']}\"\n"
        expert_string += '}\n'

        experts_answers_string += expert_string

    experts_answers_string += '\n]'

    return experts_answers_string
