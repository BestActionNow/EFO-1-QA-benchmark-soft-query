import pandas as pd
import pickle as pkl
import os
import random

def split_train_valid_test(edge_list, valid_test_ratio):
    total_num = len(edge_list)
    sample_num = int(total_num * valid_test_ratio)


    node_in_cnt, node_out_cnt = {}, {}
    for (e1, r, e2, prob) in edge_list:
        if e2 not in node_in_cnt:
            node_in_cnt[e2] = 0
        node_in_cnt[e2] += 1
        if e1 not in node_out_cnt:
            node_out_cnt[e1] = 0
        node_out_cnt[e1] += 1

    sample_cnt = 0
    train_edges = edge_list
    valid_edges, test_edges = [], []
    while sample_cnt < 2 * sample_num:
        idx =  random.sample(range(len(train_edges)), 1)[0]
        e1, r, e2, prob = train_edges[idx]
        if  node_in_cnt[e2] == 1 \
                or node_out_cnt[e1] == 1:
            continue
        if sample_cnt < sample_num:
            valid_edges.append(train_edges[idx])
        else:
            test_edges.append(train_edges[idx])
        train_edges.pop(idx)
        node_out_cnt[e1] -= 1
        node_in_cnt[e2] -= 1
        sample_cnt += 1

    return train_edges, valid_edges, test_edges


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
    raw_data_path = "data/raw/cn15k"
    processed_data_path = "data/processed/cn15k"

    df = pd.read_csv(os.path.join(raw_data_path, "entity_id.csv"), header=None, names=['ent', 'id'])
    ent2id, id2ent = {}, {}
    for idx, row in df.iterrows():
        ent, eid = row['ent'], int(row['id'])
        ent2id[ent] = eid
        id2ent[eid] = ent

    df = pd.read_csv(os.path.join(raw_data_path, "relation_id.csv"), sep=',', header=None, names=['rel', 'id'])
    rel2id, id2rel = {}, {}
    for idx, row in df.iterrows():
        rel, rid = row['rel'], int(row['id'])
        rel2id[rel] = rid
        id2rel[rid] = rel

    edge_lines = []
    with open(os.path.join(raw_data_path, "train.tsv"), 'r') as f:
        edge_lines.extend(f.readlines())
    with open(os.path.join(raw_data_path, "test.tsv"), 'r') as f:
        edge_lines.extend(f.readlines())
    edge_lines = [line.strip().split('\t') for line in edge_lines]
    edge_list = [[int(e1), int(r), int(e2), float(p)] for (e1, r, e2, p) in edge_lines]
    print("original_edge_list ", len(edge_list))
    # filter
    while True:
        node_in_cnt, node_out_cnt = {}, {}
        for (e1, r, e2, prob) in edge_list:
            if e1 not in node_out_cnt:
                node_out_cnt[e1] = 0
            node_out_cnt[e1] += 1
            if e2 not in node_in_cnt:
                node_in_cnt[e2] = 0
            node_in_cnt[e2] += 1
        need_filter = False
        for (e1, r, e2, prob) in edge_list:
            if e1 not in node_in_cnt:
                need_filter = True 
            if e2 not in node_out_cnt:
                need_filter = True
        if not need_filter or len(edge_list) == 0:
            break
        filtered_edge_list = []
        for (e1, r, e2, prob) in edge_list:
            if (e1 not in node_out_cnt) or (e2 not in node_out_cnt) or \
                (e1 not in node_in_cnt) or (e2 not in node_in_cnt):
                continue
            filtered_edge_list.append([e1,r,e2,prob])
        edge_list = filtered_edge_list
    print("filtered_edge_list: ", len(edge_list))

    # reset index
    ent_set = set([e1 for (e1, r, e2, prob) in edge_list]).union(set([e2 for (e1, r, e2, prob) in edge_list]))
    rel_set = set([r for (e1, r, e2, prob) in edge_list])
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
    for i in range(len(edge_list)):
        edge_list[i][0] = ent_old2new[edge_list[i][0]]
        edge_list[i][1] = rel_old2new[edge_list[i][1]]
        edge_list[i][2] = ent_old2new[edge_list[i][2]]
    ent2id, id2ent, rel2id, id2rel  = new_ent2id, new_id2ent, new_rel2id, new_id2rel


    valid_ratio = 0.1
    train_edges, valid_edges, test_edges = split_train_valid_test(edge_list, valid_ratio)
    print("train_edges: ", len(train_edges))
    print("test_edges: ", len(test_edges))
    print("valid_edges: ", len(valid_edges))

    dump_edges_to_file(train_edges, os.path.join(processed_data_path, 'train.txt'))
    dump_edges_to_file(valid_edges, os.path.join(processed_data_path, 'valid.txt'))
    dump_edges_to_file(test_edges, os.path.join(processed_data_path, 'test.txt'))
    dump_obj_to_file(ent2id, os.path.join(processed_data_path, 'ent2id.pkl'))
    dump_obj_to_file(id2ent, os.path.join(processed_data_path, 'id2ent.pkl'))
    dump_obj_to_file(rel2id, os.path.join(processed_data_path, 'rel2id.pkl'))
    dump_obj_to_file(id2rel, os.path.join(processed_data_path, 'id2rel.pkl'))




