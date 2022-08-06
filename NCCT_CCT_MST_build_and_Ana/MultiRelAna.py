# – coding: gbk –
import json

from Ana.DataLoader import All_the_Data_You_Need
from Ana.util import add, addSet


def main_multi_rels_entities_pair_ana(mers=None):
    if mers is None:
        mers = ['MRR']
    # 对关系实体对指标预测
    all_data = All_the_Data_You_Need(dataset='FB237', load_rrr=False, only_train_set=False)
    ht_r = {}
    trips_test = set(all_data.data['test'])
    trips_all = set(all_data.data['all_trips'])
    # trips = trips_all.difference(trips_test)
    trips = trips_all

    for h, r, t in list(trips):
        addSet((h, t), r, ht_r)
    for mer in mers:
        for model_name in ['FB237-600-full-0.5-model',
                           # 'FB237-600-full-1.0-model',
                           # 'FB237-600-full-3.0-model',
                           'FB237-600-base-model',
                           # 'FB237-600-sca-model',
                           # 'FB237-600-linear-model'
                           ]:
            patterns_mer = 0
            nums = 0
            with open('../%s_per_trips.json' % model_name, 'r', encoding='utf-8') as f:
                all_res = json.load(f)

            patterns = 'All'
            # print(len(all_data.data['test']))
            for i, hrt in enumerate(all_data.data['test']):
                h, r, t = hrt
                if (h,t) in ht_r and len(ht_r[(h,t)]) > 2:
                    patterns_mer += all_res[mer][i]
                    nums += 1

            print('model_name: %s' % model_name,
                  '三元组数量：', nums,
                  '指标 {} : {:.3f} '.format(mer, patterns_mer / (nums + 1e-9)))

if __name__ == '__main__':
    main_multi_rels_entities_pair_ana(mers=['MRR','hits@1','hits@3','hits@10','MR'])

'''
model_name: FB237-600-full-0.5-model 三元组数量： 2 指标 MRR : 0.507 
model_name: FB237-600-base-model 三元组数量： 2 指标 MRR : 0.297 
model_name: FB237-600-full-0.5-model 三元组数量： 2 指标 hits@1 : 0.500 
model_name: FB237-600-base-model 三元组数量： 2 指标 hits@1 : 0.250 
model_name: FB237-600-full-0.5-model 三元组数量： 2 指标 hits@3 : 0.500 
model_name: FB237-600-base-model 三元组数量： 2 指标 hits@3 : 0.250 
model_name: FB237-600-full-0.5-model 三元组数量： 2 指标 hits@10 : 0.500 
model_name: FB237-600-base-model 三元组数量： 2 指标 hits@10 : 0.500 
model_name: FB237-600-full-0.5-model 三元组数量： 2 指标 MR : 215.000 
model_name: FB237-600-base-model 三元组数量： 2 指标 MR : 120.250 
'''