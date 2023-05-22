from collections import defaultdict
import os.path as osp
import argparse
import os
import json

from itertools import product
from shutil import rmtree
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd

from fol.foq_v2 import (DeMorgan_replacement, concate_iu_chains, parse_formula, transform_soft_formula,
                        to_d, to_D, decompose_D, copy_query)
from formula_generation import convert_to_dnf
from utils.util import load_data_with_indexing, round_answers_value

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark_name", type=str, default="soft_fomula")
parser.add_argument("--input_formula_file", type=str, default="outputs")
parser.add_argument("--sample_size", default=10000, type=int)
parser.add_argument("--knowledge_graph", action="append", default=["ppi5k"])
parser.add_argument("--ncpus", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=50000)
parser.add_argument("--meaningful_difference_setting", type=str, default='mixed')


def normal_forms_transformation(query):
    result = {}
    # proj, rproj = load_graph()
    # query.backward_sample()
    result["original"] = query
    # result["DeMorgan"] = DeMorgan_replacement(copy_query(result["original"], True))
    # result['DeMorgan+MultiI'] = concate_iu_chains(copy_query(result["DeMorgan"], True))
    result["DNF"] = convert_to_dnf(copy_query(result["original"], True))
    # result["diff"] = to_d(copy_query(result["original"], True))
    # result["DNF+diff"] = to_d(copy_query(result["DNF"], True))
    result["DNF+MultiIU"] = concate_iu_chains(copy_query(result["DNF"], True))
    result['DNF+MultiIU'].sort_sub()
    # result["DNF+MultiIUD"] = to_D(copy_query(result["DNF+MultiIU"], True))
    # result["DNF+MultiIUd"] = decompose_D(copy_query(result["DNF+MultiIUD"], True))
    return result


def sample_1p(path, rel2percentile, easy_proj, hard_proj, data):
    #enumerate all the 1p queries
    queries_1p = []
    target_file = os.path.join(path, "train.txt")
    with open(target_file, "r",  encoding='utf-8') as f:
            for fact in tqdm(f.readlines()):
                h, r, t, p = fact.rstrip().split("\t")
                if [h, r] not in queries_1p:
                    queries_1p.append([h, r])
    count = 0
    for e, r in queries_1p:
        dobject_e = {"o": "e", "a": [int(e)]}
        dobjects_percentile = []
        for percentile in rel2percentile[r]:
            dobject_1p = {"o": "p", "a": [[int(r)], dobject_e], "f" : f"{percentile}"}
            meta_formula_v2 = json.dumps(dobject_1p)
            query_instance = parse_formula('(p-0.000,(e))')
            query_instance.additive_ground(dobject_1p)
            answer_set = query_instance.deterministic_query(easy_proj)
            data['answer_set'].append(answer_set)
            data["original"].append(query_instance.dumps)
    return data

def sample_by_row(row, easy_proj, easy_rproj, hard_proj, meaningful_difference: bool = False):
    query_instance = parse_formula(row.original)
    easy_answers = query_instance.backward_sample(easy_proj, easy_rproj, meaningful_difference=meaningful_difference)
    full_answers = query_instance.deterministic_query(hard_proj)
    hard_answers = full_answers.difference(easy_answers)
    results = normal_forms_transformation(query_instance)
    for k in results:
        assert results[k].formula == row[k]
        _full_answer = results[k].deterministic_query(hard_proj)
        assert _full_answer == full_answers
        _easy_answer = results[k].deterministic_query(easy_proj)
        assert _easy_answer == easy_answers
    return list(easy_answers), list(hard_answers), results


def sample_by_row_final(row, easy_proj, hard_proj, hard_rproj, meaningful_difference_setting: str = 'mixed'):
    while True:
        query_instance = parse_formula(row.original)
        if meaningful_difference_setting == 'mixed':
            formula = query_instance.formula
            meaningful_difference = ('d' in formula or 'D' in formula or 'n' in formula)
        elif meaningful_difference_setting == 'fixed_True':
            meaningful_difference = True
        elif meaningful_difference_setting == 'fixed_False':
            meaningful_difference = False
        else:
            assert False, 'Invalid setting!'
        full_answers = query_instance.backward_sample(hard_proj, hard_rproj,
                                                      meaningful_difference=meaningful_difference)
        assert full_answers == query_instance.deterministic_query(hard_proj)
        easy_answers = query_instance.deterministic_query(easy_proj)
        hard_answers = full_answers.difference(easy_answers)
        assert not easy_answers.intersection(hard_answers)
        results = normal_forms_transformation(query_instance)
        rounded_easy_answer = round_answers_value(list(easy_answers), 4)
        rounded_hard_answer = round_answers_value(list(hard_answers), 4)
        if set(rounded_easy_answer).intersection(rounded_hard_answer):
            easy_answers = query_instance.deterministic_query(easy_proj)
            full_answers == query_instance.deterministic_query(hard_proj)
            hard_answers = full_answers.difference(easy_answers)
            continue
        if 0 < len(hard_answers) <= 100:
            break

    # for key in results:
        # parse_formula(row[key]).additive_ground(json.loads(results[key].dumps))
    return rounded_easy_answer, rounded_hard_answer, results


def sample_by_row_final_ranking(row, proj, rproj, rel2percentile,  meaningful_difference_setting: str = 'mixed'):
    percentile_list = [25, 50, 75]
    scalr_list = [1, 0.8, 0.5]
    if "i" in row.original or "I" in row.original:
        soft_formulas = [transform_soft_formula(row.original, percentile = percentile, scalr=scalar) \
                                        for percentile, scalar in product(percentile_list, scalr_list)]
    else:
        soft_formulas = [transform_soft_formula(row.original, percentile = percentile) \
                                        for percentile in percentile_list]     
    formula_1 = soft_formulas[-1]
    while True:
        query_instance = parse_formula(formula_1)
        if meaningful_difference_setting == 'mixed':
            formula = query_instance.formula
            meaningful_difference = ('d' in formula or 'D' in formula or 'n' in formula)
        elif meaningful_difference_setting == 'fixed_True':
            meaningful_difference = True
        elif meaningful_difference_setting == 'fixed_False':
            meaningful_difference = False
        else:
            assert False, 'Invalid setting!'
        full_answers = query_instance.backward_sample(proj, rproj, rel2percentile,
                                                      meaningful_difference=meaningful_difference)
        ground_formula_1 = query_instance.formula
#        if full_answers != query_instance.deterministic_query(proj): #For debug
#            query_instance.deterministic_query(proj) 
#        assert full_answers == query_instance.deterministic_query(proj)
#        valid_answers = set([answer for answer in list(full_answers) if answer[1] > 0])
        if 0 < len(full_answers) < 200:
            break
    valid_query_object = json.loads(query_instance.dumps)
    answer_of_queries = []
    results_of_queries = []
    for fomula in soft_formulas:
            query_instance = parse_formula(fomula)
            query_instance.additive_ground(valid_query_object, rel2percentile) # add relation , entity and relation's percentile
            answer_of_queries.append(query_instance.deterministic_query(proj))
            results_of_queries.append(normal_forms_transformation(query_instance))

    # for key in results:
        # parse_formula(row[key]).additive_ground(json.loads(results[key].dumps))
    return answer_of_queries, results_of_queries


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    beta_data_folders = {"cn15k": "data/processed/cn15k",
                         "ppi5k": "data/processed/ppi5k",
                         "NELL-1115": "data/processed/NELL-1115-betae"}
    print(args.knowledge_graph)
    for kg in args.knowledge_graph:
        data_path = beta_data_folders[kg]
        ent2id, rel2id, \
            proj_train, reverse_train, \
            proj_valid, reverse_valid, \
            proj_test, reverse_test = load_data_with_indexing(data_path)

        kg_name = osp.basename(data_path).replace("-betae", "")
        out_folder = osp.join("data", args.benchmark_name, kg_name)
        with open(os.path.join(data_path, "percentile_25_50_75.json"), "r") as f:
            rel2percentile = json.load(f)
        os.makedirs(out_folder, exist_ok=True)
        modes = ["train", "valid", "test"]
        for mode in modes:
            df = pd.read_csv(os.path.join(args.input_formula_file, f"{mode}_formulas.csv"))
            for i, row in tqdm(df.iterrows(), total=len(df)):
                data = defaultdict(list)
                fid = row.formula_id
                if mode == "train":
                    if row.original == '(p,(e))':
                        data = sample_1p(data_path, rel2percentile, proj_train, proj_valid, data)
                        pd.DataFrame(data).to_csv(osp.join(out_folder, f"{mode}-{fid}.csv"), index=False)
                        number_queries_1p = len(data["answer_set"])
                        continue
                    args.num_samples = int(number_queries_1p * row.number / 1000000)
                else:
                    if row.original == '(p,(e))':
                        args.num_samples = int(number_queries_1p * 0.001)
                    else:
                        args.num_samples = row.number
                if args.ncpus > 1:
                    def sampler_func(i):
                        row_data = {}
                        easy_answers, hard_answers, results = sample_by_row_final(
                            row, proj_valid, proj_test, reverse_test,
                            meaningful_difference_setting=args.meaningful_difference_setting)
                        row_data['easy_answers'] = easy_answers
                        row_data['hard_answers'] = hard_answers
                        for k in results:
                            row_data[k] = results[k].dumps
                        return row_data

                    produced_size = 0
                    sample_size = args.num_samples
                    generated = set()
                    while produced_size < sample_size:
                        with Pool(args.ncpus) as p:
                            gets = p.map(sampler_func, list(range(sample_size - produced_size)))

                            for row_data in gets:
                                original = row_data['original']
                                if original in generated:
                                    continue
                                else:
                                    produced_size += 1
                                    generated.add(original)

                                for k in row_data:
                                    data[k].append(row_data[k])
                else:
                    generated = set()
                    sampled_query = 0
                    while sampled_query < args.num_samples:
                        if mode == "train":
                            train_answers, results = sample_by_row_final_ranking(
                                row, proj_train, reverse_test, rel2percentile, 
                                meaningful_difference_setting=args.meaningful_difference_setting)
                            if not 0 < len(train_answers):
                                continue
                        else:
                            if mode == "valid":
                                valid_answers, results = sample_by_row_final_ranking(
                                    row, proj_valid, reverse_test,rel2percentile,
                                    meaningful_difference_setting=args.meaningful_difference_setting)
                            elif mode == "test":
                                test_answers, results = sample_by_row_final_ranking(
                                    row, proj_test, reverse_test,rel2percentile,
                                    meaningful_difference_setting=args.meaningful_difference_setting)

                        if results[0]['original'].dumps in generated:
                            continue
                        else:
                            for i in range(len(results)):
                                generated.add(results[i]['original'].dumps)
                                sampled_query += 1
                                if mode == "train":
                                    data['answer_set'].append(train_answers[i])
                                elif mode == "valid":
                                    data['answer_set'].append(valid_answers[i])
                                elif mode == "valid":
                                    data['answer_set'].append(test_answers[i])
                        for k in results:
                            index = list(k.keys())[0]
                            data[index].append(k[index].dumps)

                pd.DataFrame(data).to_csv(osp.join(out_folder, f"{mode}-{fid}.csv"), index=False)
