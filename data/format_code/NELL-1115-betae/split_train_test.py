import pandas as pd
import numpy as np
import pickle as pkl
import random
import json
import copy
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--kg_raw", type=str, default="data/raw/NELL-1115-betae/NELL.08m.1115.esv.csv")
parser.add_argument("--valid_test_percen", type=float, default=0.1)
parser.add_argument("--output_path", type=str, default="data/processed/NELL-1115-betae/")

def load_nell(data_path):
    df_nell = pd.read_csv(data_path, sep='\t', error_bad_lines=False)

    # filter columns except Entity, Relation, Value, Probability
    df_nell = df_nell[['Entity', 'Relation', 'Value', 'Probability']]
    # filter rows with "generalizations" or "concept:haswikipediaurl" as Relation
    df_nell = df_nell[(df_nell['Relation'] != 'generalizations') & (df_nell['Relation'] != 'concept:haswikipediaurl')]
    # filter rows whose Entity or Value is not prefixed by "concept:"
    df_nell = df_nell[df_nell['Value'].str.startswith("concept:")]
    df_nell = df_nell[df_nell['Entity'].str.startswith("concept:")]
    df_nell = df_nell.reset_index()

    # Normalize the Probability
    prob_column = "Probability"
    # df_nell[prob_column] = (df_nell[prob_column] - df_nell[prob_column].min()) / (df_nell[prob_column].max() - df_nell[prob_column].min())

    # Get the id-name mapping of entities and relations
    ent_set = np.unique(np.concatenate([df_nell.Entity.unique(), df_nell.Value.unique()]))
    ent2id, id2ent = dict(), dict()
    for idx, entity in enumerate(ent_set):
        ent2id[entity] = idx
        id2ent[idx] = entity
    rel2id, id2rel = dict(), dict()
    for idx, relation in enumerate(df_nell.Relation.unique()):
        rel2id[relation] = idx
        id2rel[idx] = relation

    # generate edge list
    edge_list = []
    for i, row in df_nell.iterrows():
        ent1, rel, ent2, prob = row['Entity'], row['Relation'], row['Value'], row['Probability']
        ent1, rel, ent2, prob = ent2id[ent1], rel2id[rel], ent2id[ent2], float(prob)
        edge_list.append((ent1, rel, ent2, prob))

    # filter
    node_in_cnt, node_out_cnt = {}, {}
    for (e1, r, e2, prob) in edge_list:
        if e1 not in node_out_cnt:
            node_out_cnt[e1] = 0
        node_out_cnt[e1] += 1
        if e2 not in node_in_cnt:
            node_in_cnt[e2] = 0
        node_in_cnt[e2] += 1
    filtered_edge_list = []
    for (e1, r, e2, prob) in edge_list:
        if (e1 not in node_out_cnt) or (e2 not in node_out_cnt) or \
            (e1 not in node_in_cnt) or (e2 not in node_in_cnt):
            continue
        filtered_edge_list.append([e1,r,e2,prob])

    # reset index
    ent_set = set([e1 for (e1, r, e2, prob) in filtered_edge_list]).union(set([e2 for (e1, r, e2, prob) in filtered_edge_list]))
    rel_set = set([r for (e1, r, e2, prob) in filtered_edge_list])
    new_ent2id, new_id2ent, new_rel2id, new_id2rel = dict(), dict(), dict(), dict()
    ent_old2new, rel_old2new = dict(), dict()
    for i, ori_ent_id in enumerate(list(ent_set)):
        ent = id2ent[ori_ent_id]
        new_ent2id[ent] = i
        new_id2ent[i] = ent
        ent_old2new[ori_ent_id] = i
    for i, ori_rel_id in enumerate(list(rel_set)):
        rel = id2rel[ori_rel_id]
        new_rel2id[rel] = i
        new_id2rel[i] = rel
        rel_old2new[ori_rel_id] = i
    for i in range(len(filtered_edge_list)):
        filtered_edge_list[i][0] = ent_old2new[filtered_edge_list[i][0]]
        filtered_edge_list[i][1] = rel_old2new[filtered_edge_list[i][1]]
        filtered_edge_list[i][2] = ent_old2new[filtered_edge_list[i][2]]
    return filtered_edge_list, new_ent2id, new_id2ent, new_rel2id, new_id2rel



def split_train_valid_test(edge_list, valid_test_ratio):
    total_num = len(edge_list)
    sample_num = int(total_num * valid_test_ratio)


    node_link_cnt = {}
    for (e1, r, e2, prob) in edge_list:
        if e1 not in node_link_cnt:
            node_link_cnt[e1] = 0
        node_link_cnt[e1] += 1
        if e2 not in node_link_cnt:
            node_link_cnt[e2] = 0
        node_link_cnt[e2] += 1

    sample_cnt = 0
    train_edges = edge_list
    valid_edges, test_edges = [], []
    while sample_cnt < 2 * sample_num:
        idx =  random.sample(range(len(train_edges)), 1)[0]
        e1, r, e2, prob = train_edges[idx]
        if node_link_cnt[e1] == 1 or node_link_cnt[e2] == 1:
            continue
        if sample_cnt < sample_num:
            valid_edges.append(train_edges[idx])
        else:
            test_edges.append(train_edges[idx])
        train_edges.pop(idx)
        node_link_cnt[e1] -= 1
        node_link_cnt[e2] -= 1
        sample_cnt += 1

    return train_edges, valid_edges, test_edges

def generate_edge2prob(edge_list):
    edge2prob = {}
    for (e1, r, e2, prob) in edge_list:
        edge2prob[(e1, r, e2)] = prob
    return edge2prob

def dump_edges_to_file(edge_list, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'w') as f:
        for (e1, r, e2, prob) in edge_list:
            f.write("{}\t{}\t{}\t{:.4f}\n".format(e1, r, e2, prob))
    return

def dump_obj_to_file(obj, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, 'wb') as f:
        pkl.dump(obj, f)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    edge_list, ent2id, id2ent, rel2id, id2rel = load_nell(args.kg_raw)
    train_edges, valid_edges, test_edges = split_train_valid_test(edge_list, args.valid_test_percen)
    edge2prob = generate_edge2prob(edge_list)

    dump_edges_to_file(train_edges, os.path.join(args.output_path, 'train.txt'))
    dump_edges_to_file(valid_edges, os.path.join(args.output_path, 'valid.txt'))
    dump_edges_to_file(test_edges, os.path.join(args.output_path, 'test.txt'))
    dump_obj_to_file(ent2id, os.path.join(args.output_path, 'ent2id.pkl'))
    dump_obj_to_file(id2ent, os.path.join(args.output_path, 'id2ent.pkl'))
    dump_obj_to_file(rel2id, os.path.join(args.output_path, 'rel2id.pkl'))
    dump_obj_to_file(id2rel, os.path.join(args.output_path, 'id2rel.pkl'))
    dump_obj_to_file(edge2prob, os.path.join(args.output_path, 'edge2prob.pkl'))