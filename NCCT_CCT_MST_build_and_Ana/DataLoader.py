# – coding: gbk –
from Ana.util import add
import json
import os

os.environ['LOG_DIR'] = r"F:\研究生\AffE\绘图\code\logs"
os.environ['DATA_PATH'] = r"F:\研究生\AffE\绘图\code\data"
os.environ['ROOT'] = r"F:\研究生\AffE\绘图\code"
os.environ['PICTRUE_DIR'] = r"F:\研究生\AffE\绘图\code"


class All_the_Data_You_Need():
    def __init__(self, dataset='FB237',
                 path=os.environ['ROOT'] + '/data/', load_rrr=False,
                 only_train_set=False,
                 ):
        # 对关系实体对指标预测
        path = path + dataset + '/'
        self.data = {}
        self.__load(path, load_rrr=load_rrr, only_train_set=only_train_set)

    def __load(self, path, load_rrr=True, only_train_set=False):
        graph = {}
        ht_r = {}
        self.r_ht = {}
        self.hr_t = {}
        id_to_ent, ent_to_id = load_dict(os.path.join(path, 'ent.dict'))
        id_to_rel, rel_to_id = load_dict(os.path.join(path, 'rel.dict'))
        self.rel_num = len(rel_to_id)
        self.ent_num = len(ent_to_id)
        self.data['id_to_ent'] = id_to_ent
        self.data['ent_to_id'] = ent_to_id
        self.data['id_to_rel'] = id_to_rel
        self.data['rel_to_id'] = rel_to_id

        for file in ['train', 'test', 'valid']:
            file_path = os.path.join(path, file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip('\n')
                        h,r,t = line.split('\t')
                        add(file, (h,r,t), self.data)
                        if only_train_set and file == 'test':
                            continue
                        r_rev = get_rev_rel(r, self.rel_num)
                        add(h, [r, t], graph)
                        add((h, t), r, ht_r)
                        add((t, h), r_rev, ht_r)
                        add(r, [h, t], self.r_ht)
                        add(r_rev, [t, h], self.r_ht)
                        add((h, r), t, self.hr_t)
                        add((t, r_rev), h, self.hr_t)
                        add('all_trips', (h, r, t), self.data)
        all_trips_set = set(self.data['all_trips'])
        sym_rels = {}
        for h, r, t in self.data['all_trips']:
            if h != t and (t, r, h) in all_trips_set:
                if r in sym_rels:
                    sym_rels[r] += 1
                else:
                    sym_rels[r] = 1
        self.data['sym_data'] = sym_rels
        self.data['graph'] = graph
        self.data['ht_r'] = ht_r
        self.ht_r = ht_r

        if load_rrr:
            rrr_path = os.path.join(path, 'r1_r2_r3.json')
            rrr = {}
            if os.path.exists(rrr_path):
                with open(rrr_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip('\n')
                        s = line.split('\t')
                        rrr[(s[0], s[1], s[2])] = int(s[3])
                    self.data['rrr'] = rrr
            rr_ht_path = os.path.join(path, 'rr_ht.json')
            if os.path.exists(rr_ht_path):
                with open(rr_ht_path, 'r', encoding='utf-8') as f:
                    rr_ht_temp = json.load(f)
                    rr_ht = {}
                    for rr, ht in rr_ht_temp.items():
                        r1, r2 = rr.split(":<>:")
                        rr_ht[(r1, r2)] = ht
                    self.data['rr_ht'] = rr_ht


def load_dict(path):
    id_to_name = {}
    name_to_id = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            s = line.split('\t')
            id_to_name[s[0]] = s[1]
            name_to_id[s[1]] = s[0]
    return id_to_name, name_to_id


def get_rev_rel(rel_id, rel_num):
    rel_id = int(rel_id)
    if rel_id < rel_num:
        return str(rel_id + rel_num)
    return str(rel_id - rel_num)


class BuildDataDict:
    def build(self, dataset):
        path = os.environ['ROOT'] + '/data/' + dataset
        self.ent2id, self.id2ent, self.rel2id, self.id2rel = {}, {}, {}, {}
        for file in ['train', 'test', 'valid']:
            file_path = os.path.join(path, file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip('\n')
                        h, r, t = line.split('\t')
                        self.addEnt(h)
                        self.addEnt(t)
                        self.addRel(r)
        with open(os.path.join(path, "ent.dict"), 'w', encoding='utf-8') as f:
            for k, v in self.id2ent.items():
                f.write(str(k) + '\t' + str(v) + '\n')

        with open(os.path.join(path, "rel.dict"), 'w', encoding='utf-8') as f:
            for k, v in self.id2rel.items():
                f.write(str(k) + '\t' + str(v) + '\n')

    def addEnt(self, e):
        if e in self.ent2id:
            return
        self.ent2id[e] = len(self.id2ent)
        self.id2ent[len(self.ent2id)] = e

    def addRel(self, r):
        if r in self.rel2id:
            return
        self.rel2id[r] = len(self.id2rel)
        self.id2rel[len(self.rel2id)] = r