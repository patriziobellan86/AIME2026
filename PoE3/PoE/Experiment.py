import json

from tqdm import tqdm

from PoE3.PoE.FileToolkit import SaveQueriesAnswers, LoadExperts, LoadFinalDecisionMaker, get_queries, \
    LoadSingleExpertAgent, LoadExpertsFieldList, get_queries_answers
from PoE3.PoE.ModelRequests import SendToLLM, create_experts_answers_string, extract_grade, extract_confidence_score, \
    extract_reasoning_steps, extract_conclusion, extract_justification, extract_final_answer
from PoE3.PoE.prompts.experts import ASK_TO_EXPERT_USER, ASK_TO_EXPERT_SYSTEM, ASK_TO_EXPERT_NO_DESCRIPTION_SYSTEM
from PoE3.PoE.prompts.final_decision_maker import ASK_FINAL_ANSWER_USER, ASK_FINAL_ANSWER_SYSTEM
from PoE3.PoE.utilities import already_asked_to_query_expert_agents, already_asked_to_query_final_decision_maker


def AskToSingleExpert(args_dict: dict,
                      expert,
                      query=None):
    """
            This method makes a run of conversation where each expert propose its idea.
            Here there is no critique to the others. Each expert is independent.

            return answers
    """
    answers = list()
    # for expert_ID, expert in enumerate(tqdm(experts, desc="Experts processed", ascii=True)):

    messages = [{"role": "system",
                 "content": ASK_TO_EXPERT_SYSTEM.format(expert_description=expert['description'],
                                                        task=args_dict['task'],
                                                        context=args_dict['context'])},
                {"role": "user",
                 "content": ASK_TO_EXPERT_USER.format(query=query)}]

    outdict = SendToLLM(messages=messages,
                        model=args_dict['model'],
                        tokenizer=args_dict['tokenizer'],
                        device=args_dict['device'],
                        temperature=args_dict['temperature'],
                        nucleus=args_dict['nucleus'],
                        max_tokens=512,
                        )
    raw_answer = outdict['response_message']
    try:
        tmp_answer = raw_answer.replace("```json\n", "").replace("```", "")
        tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')
        tmp_dict = json.loads(tmp_answer)
        expert_final_answer = tmp_dict['final_answer']
        grade = tmp_dict['grade']
        confidence_score = tmp_dict['confidence_score']
        reasoning_steps = tmp_dict['reasoning_steps']
        justification = tmp_dict['justification']
        conclusion = tmp_dict['conclusion']

    except:
        expert_final_answer = extract_final_answer(raw_answer,
                                                   model=args_dict['model'],
                                                   tokenizer=args_dict['tokenizer'],
                                                   device=args_dict['device'],
                                                   temperature=0,
                                                   nucleus=0,
                                                   max_tokens=128,
                                                   )
        grade = extract_grade(raw_answer,
                              model=args_dict['model'],
                              tokenizer=args_dict['tokenizer'],
                              device=args_dict['device'],
                              temperature=0,
                              nucleus=0,
                              max_tokens=12,
                              )
        confidence_score = extract_confidence_score(raw_answer,
                                                    model=args_dict['model'],
                                                    tokenizer=args_dict['tokenizer'],
                                                    device=args_dict['device'],
                                                    temperature=0,
                                                    nucleus=0,
                                                    max_tokens=12,
                                                    )
        reasoning_steps = extract_reasoning_steps(raw_answer,
                                                  model=args_dict['model'],
                                                  tokenizer=args_dict['tokenizer'],
                                                  device=args_dict['device'],
                                                  temperature=0,
                                                  nucleus=0,
                                                  max_tokens=256,
                                                  )
        justification = extract_justification(raw_answer,
                                              model=args_dict['model'],
                                              tokenizer=args_dict['tokenizer'],
                                              device=args_dict['device'],
                                              temperature=0,
                                              nucleus=0,
                                              max_tokens=128,
                                              )

        conclusion = extract_conclusion(raw_answer,
                                        model=args_dict['model'],
                                        tokenizer=args_dict['tokenizer'],
                                        device=args_dict['device'],
                                        temperature=0,
                                        nucleus=0,
                                        max_tokens=128,
                                        )
    outdict['expert_ID'] = expert['expert-id']
    outdict['final_answer'] = expert_final_answer
    outdict['grade'] = grade
    outdict['confidence_score'] = confidence_score
    outdict['reasoning_steps'] = reasoning_steps
    outdict['justification'] = justification
    outdict['conclusion'] = conclusion
    return outdict


def AskToExperts(args_dict: dict,
                 experts,
                 query: str,
                 n_query: int,
                 queries_answers: dict):
    """
            This method makes a run of conversation where each expert propose its idea.
            Here there is no critique to the others. Each expert is independent.

            return answers
    """
    try:
        answers = queries_answers[str(n_query)]['experts-answers']

    except KeyError:
        answers = list()

    with tqdm(total=len(experts), desc="Experts processed", ascii=True) as ebar:
        for expert_ID, expert in enumerate(experts):
            #  check if the "expert_ID" has already answered or not
            if expert_ID <= len(answers) - 1:  # -1  since index start at 0
                ebar.set_postfix({"status": f"Expert {expert_ID} ({expert['field']}) already answered"})
                ebar.update(1)
                continue

            if 'no-description' in args_dict and args_dict['no-description']:
                expert['name'] = 'no-description'

                messages = [{"role": "system",
                             "content": ASK_TO_EXPERT_NO_DESCRIPTION_SYSTEM.format(field=expert['field'],
                                                                                   task=args_dict['task'],
                                                                                   context=args_dict['context'])}]
            else:
                #  in the case 'no-personality' is not present or is False. use the expert with personality
                messages = [{"role": "system",
                             "content": ASK_TO_EXPERT_SYSTEM.format(expert_description=expert['description'],
                                                                    task=args_dict['task'],
                                                                    context=args_dict['context'])}]
            messages.append({"role": "user",
                             "content": ASK_TO_EXPERT_USER.format(query=query)})

            outdict = SendToLLM(args_dict=args_dict,
                                messages=messages,
                                model=args_dict['model'],
                                tokenizer=args_dict['tokenizer'],
                                device=args_dict['device'],
                                temperature=args_dict['temperature'],
                                nucleus=args_dict['nucleus'],
                                max_tokens=512,
                                alternatives=args_dict['alternatives'])

            raw_answer = outdict['response_message']
            try:
                tmp_answer = raw_answer.replace("```json\n", "").replace("```", "")
                tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')
                tmp_dict = json.loads(tmp_answer)
                expert_final_answer = tmp_dict['final_answer']
                grade = tmp_dict['grade']
                confidence_score = tmp_dict['confidence_score']
                reasoning_steps = tmp_dict['reasoning_steps']
                justification = tmp_dict['justification']
                conclusion = tmp_dict['conclusion']

            except:
                expert_final_answer = extract_final_answer(args_dict=args_dict,
                                                           list_string=raw_answer,
                                                           model=args_dict['model'],
                                                           tokenizer=args_dict['tokenizer'],
                                                           device=args_dict['device'],
                                                           temperature=0,
                                                           nucleus=0,
                                                           max_tokens=128,
                                                           )
                grade = extract_grade(args_dict=args_dict,
                                      list_string=raw_answer,
                                      model=args_dict['model'],
                                      tokenizer=args_dict['tokenizer'],
                                      device=args_dict['device'],
                                      temperature=0,
                                      nucleus=0,
                                      max_tokens=12,
                                      )
                confidence_score = extract_confidence_score(args_dict=args_dict,
                                                            list_string=raw_answer,
                                                            model=args_dict['model'],
                                                            tokenizer=args_dict['tokenizer'],
                                                            device=args_dict['device'],
                                                            temperature=0,
                                                            nucleus=0,
                                                            max_tokens=12,
                                                            )
                reasoning_steps = extract_reasoning_steps(args_dict=args_dict,
                                                          list_string=raw_answer,
                                                          model=args_dict['model'],
                                                          tokenizer=args_dict['tokenizer'],
                                                          device=args_dict['device'],
                                                          temperature=0,
                                                          nucleus=0,
                                                          max_tokens=256,
                                                          )
                justification = extract_justification(args_dict=args_dict,
                                                      list_string=raw_answer,
                                                      model=args_dict['model'],
                                                      tokenizer=args_dict['tokenizer'],
                                                      device=args_dict['device'],
                                                      temperature=0,
                                                      nucleus=0,
                                                      max_tokens=128,
                                                      )

                conclusion = extract_conclusion(args_dict=args_dict,
                                                list_string=raw_answer,
                                                model=args_dict['model'],
                                                tokenizer=args_dict['tokenizer'],
                                                device=args_dict['device'],
                                                temperature=0,
                                                nucleus=0,
                                                max_tokens=128,
                                                )
            outdict['expert-id'] = expert_ID
            outdict['final_answer'] = expert_final_answer
            outdict['grade'] = grade
            outdict['confidence_score'] = confidence_score
            outdict['reasoning_steps'] = reasoning_steps
            outdict['justification'] = justification
            outdict['conclusion'] = conclusion

            answers.append(outdict)
            experts_answers = {"query": query,
                               "experts-answers": answers}

            try:
                queries_answers[str(n_query)].update(experts_answers)
            except KeyError:
                queries_answers[str(n_query)] = experts_answers
            SaveQueriesAnswers(args_dict, queries_answers)
            ebar.set_postfix({"status": f"Expert {expert_ID} {expert['name']} ({expert['field']}) answer saved"})
            ebar.update(1)

    return answers


def QueriesToExpertAgents(args_dict):
    #  Load experts

    if 'no-description' in args_dict and args_dict['no-description']:
        # load expertize fields
        expertizes = LoadExpertsFieldList(args_dict)
        # rename the dict fields from 'list' to 'field'
        experts = list()
        for field in expertizes:
            experts.append({"field": field})
    else:
        #  load expert personalities
        experts = LoadExperts(args_dict)

    #  Load queries and answers
    queries = get_queries(args_dict)
    queries_answers = get_queries_answers(args_dict)

    # print(f"len queries: {len(queries)}, query_last_index: {queries_last_index} len agents: {len(experts)}")
    # for n_query, query in enumerate(tqdm(queries[queries_last_index:], desc="Queries processed", ascii=True)):
    with tqdm(total=len(queries), desc="Queries processed", ascii=True) as pbar:
        for n_query, query in enumerate(queries):
            # print(f"already_asked_to_query_expert_agents(queries_answers, n_query)? {already_asked_to_query_expert_agents(queries_answers, n_query)}")
            if already_asked_to_query_expert_agents(queries_answers, n_query, experts):
                pbar.set_postfix({"status": f"Query {n_query} already asked"})
                pbar.update(1)
                continue
            query_answers = AskToExperts(args_dict=args_dict,
                                         experts=experts,
                                         query=query,
                                         n_query=n_query,
                                         queries_answers=queries_answers)

            experts_answers = {"query": query,
                               "experts-answers": query_answers}
            try:
                queries_answers[str(n_query)].update(experts_answers)
            except KeyError:
                queries_answers[str(n_query)] = experts_answers

            SaveQueriesAnswers(args_dict, queries_answers)

            # print(f"Query {n_query} completed with {len(query_answers)} queries")
            pbar.set_postfix({"status": f"Query {n_query} saved"})
            pbar.update(1)

    return queries_answers


def QueriesToASingleExpertAgent(args_dict, expert_id):
    #  read the queries
    queries = get_queries(args_dict)
    queries_answers = get_queries_answers(args_dict)

    expert = LoadSingleExpertAgent(args_dict, expert_id)
    raise NotImplementedError("Experiment.py")
    for n_query, query in enumerate(tqdm(queries, desc="Queries processed", ascii=True)):
        if already_asked_to_query_single_expert_agent(queries_answers, n_query):
            print(f"Query {n_query} already asked to the baseline model")
            continue
        raise NotImplementedError("This method needs to be fixed")

        query_answers = AskToSingleExpert(args_dict=args_dict,
                                          expert=expert,
                                          query=query)

        queries_answers[n_query] = {"query": query,
                                    "query_answers": query_answers}

        SaveQueriesAnswers(args_dict, queries_answers)


def QueriesToMakeFinalDecision(args_dict):
    #  load agents
    if 'no-description' in args_dict and args_dict['no-description']:
        # load expertize fields
        expertizes = LoadExpertsFieldList(args_dict)
        # rename the dict fields from 'list' to 'field'
        experts = list()
        for field in expertizes:
            experts.append({"field": field})
    else:
        #  load expert personalities
        experts = LoadExperts(args_dict)

    final_decision_maker = LoadFinalDecisionMaker(args_dict)

    #  read the queries and answers files
    queries = get_queries(args_dict)
    queries_answers = get_queries_answers(args_dict)
    for n_query, query in enumerate(queries_answers):
        if not already_asked_to_query_expert_agents(queries_answers, n_query, experts):
            print(f"Query {n_query} NOT! asked")
            raise ValueError("You must ask to expert agents first")
    print("all the queries are done by agents. proceeding with the final decision maker.")
    for n_query, query in enumerate(tqdm(queries, desc="Queries processed", ascii=True)):
        if already_asked_to_query_final_decision_maker(queries_answers, n_query):
            # print(f"Query {n_query} already asked to the Final Decision Maker agent")
            continue

        # if already_asked_to_query_expert_agents(queries_answers, n_query, experts):
        query_answers = queries_answers[str(n_query)]["experts-answers"]
        # else:
        #     raise ValueError(f"Experts did not respond to query {n_query}. run query to expert agents before making final decision.")

        outdict_fdm = AskToFinalDecisionMaker(
            args_dict,
            final_decision_maker=final_decision_maker,
            experts=experts,
            query=query,
            query_answers=query_answers)
        #  update results in the dictionary
        queries_answers[str(n_query)].update(outdict_fdm)
        SaveQueriesAnswers(args_dict, queries_answers)


def AskToFinalDecisionMaker(args_dict: dict,
                            final_decision_maker: dict,
                            experts: list,
                            query: str,
                            query_answers: list):
    experts_answers = create_experts_answers_string(query_answers, experts)
    messages = [{"role": "system",
                 "content": ASK_FINAL_ANSWER_SYSTEM.format(
                     description=final_decision_maker['description'],
                     task=args_dict['task'],
                     context=args_dict['context'],
                 )
                 }, {"role": "user", "content": ASK_FINAL_ANSWER_USER.format(query=query,
                                                                             experts_answers=experts_answers,
                                                                             )
                     }]

    outdict = SendToLLM(args_dict=args_dict,
                        messages=messages,
                        model=args_dict['model'],
                        tokenizer=args_dict['tokenizer'],
                        device=args_dict['device'],
                        temperature=args_dict['temperature'],
                        nucleus=args_dict['nucleus'],
                        max_tokens=1024)
    raw_answer = outdict['response_message']
    outdict_fdm = {f"final-decision-maker-{k}": v for k, v in outdict.items()}

    try:
        tmp_answer = raw_answer.replace("```json\n", "").replace("```", "")
        tmp_answer = tmp_answer.replace("\n", "").replace('\\', '')
        tmp_dict = json.loads(tmp_answer)
        reasoning_steps = tmp_dict['reasoning_steps']
        conclusion = tmp_dict['conclusion']
        final_answer = tmp_dict['final_answer']

    except:
        #  using LLM strategy

        reasoning_steps = extract_reasoning_steps(args_dict,
                                                  raw_answer,
                                                  model=args_dict['model'],
                                                  tokenizer=args_dict['tokenizer'],
                                                  device=args_dict['device'],
                                                  temperature=0,
                                                  nucleus=0,
                                                  max_tokens=512,
                                                  )

        conclusion = extract_conclusion(args_dict,
                                        raw_answer,
                                        model=args_dict['model'],
                                        tokenizer=args_dict['tokenizer'],
                                        device=args_dict['device'],
                                        temperature=0,
                                        nucleus=0,
                                        max_tokens=256,
                                        )
        final_answer = extract_final_answer(args_dict,
                                            raw_answer,
                                            model=args_dict['model'],
                                            tokenizer=args_dict['tokenizer'],
                                            device=args_dict['device'],
                                            temperature=0,
                                            nucleus=0,
                                            max_tokens=256,
                                            )

    outdict_fdm['final-decision-maker-answer'] = final_answer
    outdict_fdm['final-decision-maker-reasoning-steps'] = reasoning_steps
    outdict_fdm['final-decision-maker-conclusion'] = conclusion
    return outdict_fdm
