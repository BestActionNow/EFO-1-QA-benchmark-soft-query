from fol.foq_v2 import *
from itertools import product
from utils.util import load_data_with_indexing
import os

def transform_beta_formula(fosq_formula: str, percentile=0, scalr=1):
    # Transform  query fomulas of beraEto its soft_fosq_formula
    cached_objects = {}
    cached_subranges = {}
    todo_ranges = []

    def identify_range(i, j):
        """ i, and j is the index of ( and ) respectively
        identify the information contained in the range
        return
            ops: operational string
            sub_range_list: a list of sub ranges
        """
        ops = fosq_formula[i + 1]
        level_stack = []
        sub_range_list = []
        for k in range(i + 1, j):
            if fosq_formula[k] == '(':
                level_stack.append(k)
            elif fosq_formula[k] == ')':
                begin = level_stack.pop(-1)
                if len(level_stack) == 0:
                    sub_range_list.append((begin, k))
        if ops == 'e':
            assert len(sub_range_list) == 0
        elif ops in 'pn':
            assert len(sub_range_list) == 1
        elif ops == 'd':
            assert len(sub_range_list) == 2
        elif ops in 'uiIUD':
            assert len(sub_range_list) > 1
        elif ops in '()':
            return identify_range(i + 1, j - 1)
        else:
            raise NotImplementedError(f"Ops {ops} is not defined")
        return ops, sub_range_list


    _b = 0
    _e = len(fosq_formula) - 1
    todo_ranges.append((_b, _e))
    while (_b, _e) not in cached_objects:
        i, j = todo_ranges[-1]

        if (i, j) in cached_subranges:
            ops, sub_range_list = cached_subranges[(i, j)]
        else:
            ops, sub_range_list = identify_range(i, j)
            cached_subranges[(i, j)] = (ops, sub_range_list)

        valid_sub_ranges = True
        for _i, _j in sub_range_list:
            if not (_i, _j) in cached_objects:
                todo_ranges.append((_i, _j))
                valid_sub_ranges = False

        if valid_sub_ranges is True:
            args = [cached_objects[r] for r in sub_range_list]
            if ops == "e":
                obj = "(e)"
            elif ops == "p":
                obj = f"(p-{percentile},{args[0]})"
            elif ops == "i":
                imporatnce = "".join([f"-{scalr**i}" for i in range(len(args))])
                sub_soft_formula = "".join([f",{args[i]}" for i in range(len(args))])
                obj = f"(i{imporatnce}{sub_soft_formula})"
            elif ops == "n":
                obj = f"(n,{args[0]})"
            elif ops == "u":
                sub_soft_formula = "".join([f"-{args[i]}" for i in range(len(args))])
                obj = f"(u,{sub_soft_formula})"
            else:
                print("Ddin't support this operation!")
#            obj = ops_dict[ops](*args)
            todo_ranges.pop(-1)
            cached_objects[(i, j)] = obj
    return cached_objects[_b, _e]

def parse_formula_original(fosq_formula: str) -> FirstOrderSetQuery:
    """ A new function to parse first-order set query string
    """
    cached_objects = {}
    cached_subranges = {}
    todo_ranges = []

    def identify_range(i, j):
        """ i, and j is the index of ( and ) respectively
        identify the information contained in the range
        return
            ops: operational string
            sub_range_list: a list of sub ranges
        """
        ops = fosq_formula[i + 1]
        level_stack = []
        sub_range_list = []
        for k in range(i + 1, j):
            if fosq_formula[k] == '(':
                level_stack.append(k)
            elif fosq_formula[k] == ')':
                begin = level_stack.pop(-1)
                if len(level_stack) == 0:
                    sub_range_list.append((begin, k))
        if ops == 'e':
            assert len(sub_range_list) == 0
        elif ops in 'pn':
            assert len(sub_range_list) == 1
        elif ops == 'd':
            assert len(sub_range_list) == 2
        elif ops in 'uiIUD':
            assert len(sub_range_list) > 1
        elif ops in '()':
            return identify_range(i + 1, j - 1)
        else:
            raise NotImplementedError(f"Ops {ops} is not defined")
        return ops, sub_range_list

    _b = 0
    _e = len(fosq_formula) - 1
    todo_ranges.append((_b, _e))
    while (_b, _e) not in cached_objects:
        i, j = todo_ranges[-1]

        if (i, j) in cached_subranges:
            ops, sub_range_list = cached_subranges[(i, j)]
        else:
            ops, sub_range_list = identify_range(i, j)
            cached_subranges[(i, j)] = (ops, sub_range_list)

        valid_sub_ranges = True
        for _i, _j in sub_range_list:
            if not (_i, _j) in cached_objects:
                todo_ranges.append((_i, _j))
                valid_sub_ranges = False

        if valid_sub_ranges is True:
            sub_objects = [cached_objects[r] for r in sub_range_list]
            obj = ops_dict[ops](*sub_objects)
            todo_ranges.pop(-1)
            cached_objects[(i, j)] = obj
    return cached_objects[_b, _e]



ent2id, rel2id, \
        proj_train, reverse_train, \
        proj_valid, reverse_valid, \
        proj_test, reverse_test = load_data_with_indexing("data/processed/ppi5k")
beta_formula = ["(i,(p,(e)),(p,(p,(e))))"]
dump = '{"o": "i", "a": [{"o": "n", "a": {"o": "p", "a": [[0], {"o": "e", "a": [91]}], "f": "0.560"}}, {"o": "p", "a": [[0], {"o": "e", "a": [2059]}], "f": "0.560"}], "f": "0.323-0.677"}'
percentile_list = [25, 50, 75]
scalr_list = [1, 0.8, 0.5]

with open(os.path.join("data/processed/ppi5k", "percentile_25_50_75.json"), "r") as f:
    rel2percentile = json.load(f)

for fomula in beta_formula:
    while True:
        if "i" in fomula:
            soft_formulas = [transform_beta_formula(fomula, percentile = percentile, scalr=scalar) for percentile, scalar in product(percentile_list, scalr_list)]
        else:
            soft_formulas = [transform_beta_formula(fomula, percentile = percentile) for percentile in percentile_list]
        formula_1 = soft_formulas[-1]
        query_instance = parse_formula(formula_1)
        while True:
            answer = query_instance.backward_sample(proj_train, reverse_train,rel2percentile,
                                                        meaningful_difference=False)
            
            if len(answer) > 0:
                break
            
        valid_query_object = json.loads(query_instance.dumps)
        queries = []
        for fomula in soft_formulas:
            query_instance = parse_formula(fomula)
            query_instance.additive_ground(valid_query_object, rel2percentile)
            real_answer = query_instance.deterministic_query(proj_train)
            queries.append(query_instance)
        real_answer == answer
        query_instance.deterministic_query(proj_train)