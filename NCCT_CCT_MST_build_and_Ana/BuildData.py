# – coding: gbk –
import json

from Ana.DataLoader import All_the_Data_You_Need
from Ana.util import add, find_head_and_tail
import os

DATA_PATH = ".data"

def find_comm_com_examples(dataset = 'FB237'):
    all_data = All_the_Data_You_Need(dataset=dataset,load_rrr=False)
    # entity_name_to_key = {}
    # e2w = load_entity2wikidata('data/FB237/entity2wikidata.json')
    # for k,v in e2w.items():
    #    entity_name_to_key[v['label']] = k
    # 找到所有多重关系

    # 输出所有满足
    hrr_ts = find_head_and_tail(all_data.data['graph'])
    rr_r = {}
    rr_ht = {}
    for hrr, ts in hrr_ts.items():
        h, r1, r2 = hrr
        for t in ts:
            add(r1 + ':<>:' + r2, (h, t), rr_ht)
            if (h, t) in all_data.data['ht_r']:
                for r3 in all_data.data['ht_r'][(h, t)]:
                    if (r1, r2, r3) in rr_r:
                        rr_r[(r1, r2, r3)] += 1
                    else:
                        rr_r[(r1, r2, r3)] = 1
    with open(os.path.join(DATA_PATH,dataset,'r1_r2_r3.json'), 'w', encoding='utf-8') as f:
        for rrr, n in rr_r.items():
            f.write('%s\t%s\t%s\t%d\n' % (rrr[0], rrr[1], rrr[2], n))

    with open(os.path.join(DATA_PATH,dataset,'rr_ht.json'), 'w', encoding='utf-8') as f:
        json.dump(rr_ht, f)


if __name__ == '__main__':
    find_comm_com_examples('FB237')
