import tqdm
#graph_paths = ["data/processed/cn15k", "data/processed/ppi5k"]
graph_paths = ["data/FB15k", "data/FB15k-237", "data/NELL"]

#Target: get the entity number, relation number
for graph_path in graph_paths:
    files = ["train"]
    for file in files:
        relations = []
        entities = []
        queries_1p = []
        target_file = graph_path + "/" + file + ".txt"
        with open(target_file, "r",  encoding='utf-8') as f:
            for fact in tqdm.tqdm(f.readlines()):
                h, r, t = fact.rstrip().split("\t")
                if r not in relations:
                    relations.append(r)
                if h not in entities:
                    entities.append(h)
                if t not in entities:
                    entities.append(t)
                if [h, r] not in queries_1p:
                    queries_1p.append([h, r])

        print(f"the number of entities is {len(entities)}")
        print(f"the number of relations is {len(relations)}")
        print(f"the number of 1p queries is {len(queries_1p)}")