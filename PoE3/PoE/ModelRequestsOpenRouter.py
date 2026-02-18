from prompts.tools_instructions import CLEAN_LIST_SYSTEM, CLEAN_LIST_USER, EXTRACT_BASE_USER, \
    EXTRACT_NAME_SYSTEM, EXTRACT_DESCRIPTION_SYSTEM, EXTRACT_JUSTIFICATION_SYSTEM, \
    EXTRACT_GRADE_SYSTEM, EXTRACT_FINAL_ANSWER_SYSTEM, \
    EXTRACT_CONFIDENCE_SCORE_SYSTEM, EXTRACT_REASONING_STEPS_SYSTEM, EXTRACT_CONCLUSION_SYSTEM
#
# from prompts.chat_template import chat_template

from typing import Tuple
from time import time
import numpy as np
from nltk import sent_tokenize
import requests
import json

seed_value = 23


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

    apikey = open('apikey', 'r').read().strip()
    key = f"Bearer {apikey}"

    start_time = time()
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        # url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": key,
            "Content-Type": "application/json",
            # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
            # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": "meta-llama/llama-3.1-70b-instruct",

            "logprobs": True,
            'seed': 245,

            "max_tokens": 1215,

            "temperature": 0.7,
            "top_p": 0.9,
            "min_p": 0.1,
            "top_a": 0.9,

            "messages": [
                {
                    "role": "user",
                    "content": "Perchè il cielo è grigio sopra Torino?"
                }
            ],

            "usage": True,  # get usage information

            "provider": {

                "data_collection": "deny",  # do not share data with the provider and do not use it for training
                "only": ["together",  # list of providers to use
                         "deepinfra",
                         "fireworks"],
                'allow_fallbacks': False,
                # do not allow fallback to other providers if the selected ones are not available
                "sort": "price",  # sort providers by price

                'quantizations': [
                    'fp4', 'fp8', 'fp32'
                ]
            }
        })
    )

    class PostCompletion:
        def __init__(self, response):
            self.response = response
            self.text_response_json = json.loads(self.response.text)

        def get_content(self):
            return self.text_response_json['choices'][0]['message']['content']

        def get_message_role(self):
            return self.text_response_json['choices'][0]['message']['role']

        def get_metadata(self):
            return {
                "id": self.text_response_json['id'],
                "elapsed_time": self.response.elapsed.total_seconds(),
                "model": self.text_response_json['model'],
                "provider_name": self.text_response_json['provider'],
                "finish_reason": self.text_response_json['choices'][0]['finish_reason'],
                "prompt_tokens": self.text_response_json['usage']['prompt_tokens'],
                "completion_tokens": self.text_response_json['usage']['completion_tokens'],
            }

    print(PostCompletion(response).get_content())
    print(PostCompletion(response).get_message_role())
    print(PostCompletion(response).get_metadata())






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


def update_messages(messages: list,
                    role: str,
                    query: str,
                    ):
    messages.append({"role": role,
                     "content": query})
    return messages


def clean_list(list_string: str,
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
    outdict = SendToLLM(messages=messages,
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


def extract_base(base_prompt: str,
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
    outdict = SendToLLM(messages=messages,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens)
    return outdict["response_message"]


def extract_name(list_string: str,
                 model=None,
                 tokenizer=None,
                 device=None,
                 temperature=1.2,
                 nucleus=0.0,
                 max_tokens=64) -> str:
    return extract_base(base_prompt=EXTRACT_NAME_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_description(list_string: str,
                        model=None,
                        tokenizer=None,
                        device=None,
                        temperature=1.2,
                        nucleus=0.0,
                        max_tokens=4096) -> str:
    return extract_base(base_prompt=EXTRACT_DESCRIPTION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_grade(list_string: str,
                  model=None,
                  tokenizer=None,
                  device=None,
                  temperature=1.2,
                  nucleus=0.0,
                  max_tokens=32) -> str:
    return extract_base(base_prompt=EXTRACT_GRADE_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_justification(list_string: str,
                          model=None,
                          tokenizer=None,
                          device=None,
                          temperature=1.2,
                          nucleus=0.0,
                          max_tokens=248) -> str:
    return extract_base(base_prompt=EXTRACT_JUSTIFICATION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_final_answer(list_string: str,
                         model=None,
                         tokenizer=None,
                         device=None,
                         temperature=1.2,
                         nucleus=0.0,
                         max_tokens=1024) -> str:
    return extract_base(base_prompt=EXTRACT_FINAL_ANSWER_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_confidence_score(list_string: str,
                             model=None,
                             tokenizer=None,
                             device=None,
                             temperature=1.2,
                             nucleus=0.0,
                             max_tokens=16) -> str:
    return extract_base(base_prompt=EXTRACT_CONFIDENCE_SCORE_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def extract_reasoning_steps(list_string: str,
                            model=None,
                            tokenizer=None,
                            device=None,
                            temperature=1.2,
                            nucleus=0.0,
                            max_tokens=2048) -> list:
    outdict = extract_base(base_prompt=EXTRACT_REASONING_STEPS_SYSTEM,
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


def extract_conclusion(list_string: str,
                       model=None,
                       tokenizer=None,
                       device=None,
                       temperature=1.2,
                       nucleus=0.0,
                       max_tokens=1024) -> str:
    return extract_base(base_prompt=EXTRACT_CONCLUSION_SYSTEM,
                        string=list_string,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        temperature=temperature,
                        nucleus=nucleus,
                        max_tokens=max_tokens,
                        )


def create_experts_answers_string(query_answers: list, experts: list) -> str:
    experts_answers_string = '[\n'
    for ans, expert in zip(query_answers, experts):
        expert_string = "{\n"
        expert_string += f"\"expert-name\": \"{expert['name']}\",\n"
        expert_string += f"\"expert-field\": \"{expert['field']}\",\n"
        expert_string += f"\"answer\": \"{ans['final_answer']}\",\n"
        expert_string += f"\"grade\": \"{ans['grade']}\",\n"
        expert_string += f"\"confidence-score\": \"{ans['confidence-score']}\",\n"
        expert_string += f"\"justification\": \"{ans['justification']}\",\n"
        expert_string += f"\"reasoning-steps\": \"{ans['reasoning-steps']}\",\n"
        expert_string += f"\"conclusion\": \"{ans['conclusion']}\"\n"
        expert_string += '}\n'

        experts_answers_string += expert_string
    # print(f"{experts_answers_string=}")

    experts_answers_string += '\n]'

    return experts_answers_string
