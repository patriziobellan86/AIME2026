import codecs
import json
import datetime
from tqdm import tqdm

from PoE3.PoE.FileToolkit import get_queries,  SaveQueriesAnswers
from PoE3.PoE.FileToolkit import get_queries_answers
from PoE3.PoE.ModelRequests import SendToLLM
from PoE3.PoE.utilities import already_asked_to_query_baseline


def RunBaseline(args_dict):
    """
    Run the baseline model on the queries and save the answers
    :param args_dict:
    :return:
    """
    queries = get_queries(args_dict)
    # queries_answers, queries_last_index= get_queries_answers_for_baselines(args_dict)
    queries_answers = get_queries_answers(args_dict)
    with tqdm(total=len(queries)) as pbar:
        for n_query, query in enumerate(tqdm(queries, desc="Queries processed", ascii=True)):
            #  go to the right index
            # n_query += queries_last_index
            if already_asked_to_query_baseline(queries_answers, n_query):
                # print(f"Query {n_query} already asked to the baseline model")
                # print(f"Query {n_query} already asked to the baseline model")
                pbar.update(1)
                pbar.set_description(f"Query {n_query} already asked to the baseline model")
                continue

            messages = list()
            query_str = f"{args_dict['task']} in the context of: {args_dict['context']}. {query}"
            messages.append({"role": "user", "content": query_str})
            # Get the current date and time
            now = datetime.datetime.now()

            # print("send request to llm", now)
            outdict = SendToLLM(args_dict=args_dict,
                                messages=messages,
                                model=args_dict['model'],
                                tokenizer=args_dict['tokenizer'],
                                device=args_dict['device'],
                                temperature=args_dict['temperature'],
                                nucleus=args_dict['nucleus'],
                                max_tokens=1024,
                                alternatives=args_dict['alternatives'])
            #  rename keys
            # Get the current date and time
            now = datetime.datetime.now()

            # print("query processed:", now)
            baseline_outdict = {f"baseline-{k}": v for k, v in outdict.items()}
            baseline_outdict["baseline-answer"] = baseline_outdict["baseline-response_message"]
            #  remove old key
            baseline_outdict.pop("baseline-response_message")
            #  add query
            baseline_outdict['query'] = query
            #  update results in the dictionary
            try:
                queries_answers[n_query].update(baseline_outdict)
            except KeyError:
                queries_answers[n_query] = baseline_outdict

            SaveQueriesAnswers(args_dict, queries_answers)
            # print(f"Query {n_query} saved")
            pbar.update(1)
            pbar.set_description(f"Query {n_query} saved")

    return queries_answers
