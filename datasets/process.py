"""Knowledge Graph dataset pre-processing functions."""

import collections
import os
import pickle

import numpy as np
import os

os.environ['LOG_DIR'] = "./logs"
os.environ['DATA_PATH'] = "./data"

def get_idx(path, splits = None):
    """Map entities and relations to unique ids.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """
    entities, relations = set(), set()
    if splits is None:
        splits = ["train", "valid", "test"]
    for split in splits:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}
    return ent2idx, rel2idx


def to_np_array(dataset_file, ent2idx, rel2idx):
    """Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")

# 添加对称边标识集
def symm_rel_id(examples,r_num):
    temp_set = set([(h,r,t) for h,r,t in examples])
    relids = [0. for i in range(r_num)]
    for h, r, t in examples:
        if (t, r, h) in temp_set:
            relids[r] = 1.
    return np.array(relids).astype('float32')

def get_filters(examples, n_relations):
    """Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final


def process_dataset(path, splits = None):
    """Map entities and relations to ids and saves corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    if splits is None:
        splits = ["train", "valid", "test"]
    ent2idx, rel2idx = get_idx(path, splits)
    examples = {}
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}

    # 添加对称边标识集
    sym_rel_ids = symm_rel_id(examples['train'],len(rel2idx))
    return examples, filters,sym_rel_ids


if __name__ == "__main__":
    '''
        data_path = os.environ["DATA_PATH"]
        for dataset_name in os.listdir(data_path):
            dataset_path = os.path.join(data_path, dataset_name)
            dataset_examples, dataset_filters, sym_rel_ids = process_dataset(dataset_path)
            for dataset_split in ["train", "valid", "test"]:
                save_path = os.path.join(dataset_path, dataset_split + ".pickle")
                with open(save_path, "wb") as save_file:
                    pickle.dump(dataset_examples[dataset_split], save_file)
            with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
                pickle.dump(dataset_filters, save_file)
    
            # 添加对称边标识集
            with open(os.path.join(dataset_path, "sym_rel_ids.pickle"), "wb") as save_file:
                pickle.dump(sym_rel_ids, save_file)
    '''
    data_path = '../data/'
    dataset_name = 'Non-CommCom'
    dataset_path = os.path.join(data_path, dataset_name)
    dataset_splits = ["train", "valid", "test"]
    dataset_examples, dataset_filters, sym_rel_ids = process_dataset(dataset_path, dataset_splits)
    for dataset_split in dataset_splits:
        save_path = os.path.join(dataset_path, dataset_split + ".pickle")
        with open(save_path, "wb") as save_file:
            pickle.dump(dataset_examples[dataset_split], save_file)
    with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
        pickle.dump(dataset_filters, save_file)