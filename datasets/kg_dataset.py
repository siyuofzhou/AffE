"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl

import numpy as np
import torch


class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug, splits=None):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        if splits is None:
            splits = ['train','valid','test']
        else:
            if "train" not in splits:
                splits.append("train")
                
        for split in splits:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")

        # 对称关系id
        # sym_rel_ids = open(os.path.join(self.data_path, "sym_rel_ids.pickle"), "rb")
        # self.sym_rel_ids = pkl.load(sym_rel_ids).reshape([-1,1])
        # sym_rel_ids.close()

        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2

    def insert(self, node_rels, h, r):
        if h in node_rels:
            if r in node_rels[h]:
                node_rels[h][r] += 1
            else:
                node_rels[h][r] = 1
        else:
            node_rels[h] = {}
            node_rels[h][r] = 1

    def insert_rt(self, node_rels, node_tails, h, r, t):
        if h in node_rels:
            node_rels[h].append(r)
            node_tails[h].append(t)
        else:
            node_rels[h] = [r]
            node_tails[h] = [t]

    def get_node_r_t(self, split, topk=10):
        examples = self.data[split]
        topk_node_rels = np.zeros([self.n_entities, topk])
        topk_node_tails = np.zeros([self.n_entities, topk])
        topk_node_mask = np.zeros([self.n_entities,topk])
        node_rels = {}
        node_tails = {}
        for h, r, t in examples:
            self.insert_rt(node_rels, node_tails, h, r + self.n_predicates // 2, t)
            self.insert_rt(node_rels, node_tails, t, r, h)
        for k,rels in node_rels.items():
            tails = node_tails[k]
            n = 0
            for r,t in zip(rels,tails):
                topk_node_tails[k][n] = t
                topk_node_rels[k][n] = r
                topk_node_mask[k][n] = 1.0
                n += 1
                if n == topk:
                    break
        return torch.from_numpy(topk_node_rels.astype("int64")), \
               torch.from_numpy(topk_node_tails.astype("int64")),\
               torch.from_numpy(topk_node_mask.astype("float32"))

    def get_node_rels(self, split, topk=10):
        # 获取节点周围边的id和权重
        examples = self.data[split]
        node_rels = {}
        for h,r,t in examples:
            self.insert(node_rels, h, r)
            self.insert(node_rels, t, r + self.n_predicates // 2)
        topk_node_rels = np.zeros([self.n_entities, topk],dtype=np.int64)
        topk_weight = np.zeros([self.n_entities, topk],dtype=np.float32)
        for node, rs in node_rels.items():
            _rs = sorted(rs.items(), key=lambda x:x[1], reverse=True)
            if len(_rs) >= topk:
                ws = [w[1] for w in _rs[:topk]]
                all_w = sum(ws)
                for i, item in enumerate(_rs[:topk]):
                    r, c = item
                    topk_node_rels[node][i] = r
                    topk_weight[node][i] = c/all_w
            else:
                cs = [c for r,c in _rs]
                all_w = sum(cs)
                for i,item in enumerate(_rs):
                    r, c = item
                    topk_node_rels[node][i] = r
                    topk_weight[node][i] = c / all_w
        return torch.from_numpy(topk_node_rels.astype("int64")), torch.from_numpy(topk_weight.astype("float32"))



    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
