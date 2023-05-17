import tqdm
import json
import os
import numpy as np
graph_paths = ["data/processed/ppi5k", "data/processed/cn15k"]
#graph_paths = ["data/FB15k", "data/FB15k-237", "data/NELL"]

#Target: get the entity number, relation number
for graph_path in graph_paths:
    files = ["train"]
    for file in files:
        rel2uncertain = {}
        target_file = graph_path + "/" + file + ".txt"
        with open(target_file, "r",  encoding='utf-8') as f:
            for fact in tqdm.tqdm(f.readlines()):
                h, r, t, p = fact.rstrip().split("\t")
                if int(r) not in rel2uncertain:
                    rel2uncertain[int(r)] = [float(p)]
                else:
                    rel2uncertain[int(r)].append(float(p))
        rel2percentile = {}
        for rel in sorted(rel2uncertain.keys()):
            pre_25 = np.percentile(rel2uncertain[rel], 25)
            pre_50 = np.percentile(rel2uncertain[rel], 50)
            pre_75 = np.percentile(rel2uncertain[rel], 75)
            rel2percentile[rel] = [pre_25, pre_50, pre_75]
        with open(os.path.join(graph_path, 'percentile_25_50_75.json'), 'w') as f:
            json.dump(rel2percentile, f)
#        print(f"the number of 1p queries is {len(queries_1p)}")