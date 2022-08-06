# – coding: gbk –
import json

from Ana.DataLoader import All_the_Data_You_Need, get_rev_rel
from Ana.util import add


def add_set(k, v, dic):
    if k in dic:
        dic[k].add(v)
    else:
        dic[k] = {v}


def main_composition_ana(mers=None,
                         comm_com=False,  # 可交换组合补全
                         top_num_value=500  # 组合模式中r1r2->r3出现频率排名第 top_num_value的值
                         ):
    if mers is None:
        mers = ['MRR']
    # 对关系实体对指标预测
    all_data = All_the_Data_You_Need(dataset='FB237', load_rrr=True)
    trips_comm = set()
    trips_non_comm = set()


    values = [n for rrr, n in all_data.data['rrr'].items()]
    values = sorted(values)
    up = values[-top_num_value]
    print(up)
    rrr_temp = set()
    for rrr, n in all_data.data['rrr'].items():
        r1, r2, r3 = rrr
        if r1 in all_data.data["sym_data"] or r2 in all_data.data["sym_data"] or r3 in all_data.data["sym_data"]:
            continue
        if n < up:
            continue
        rrr_temp.add((r1, r2, r3))
    print("data OK, rrr_temp len",len(rrr_temp))
    rr_r = {}
    for r1, r2, r3 in rrr_temp:
        add_set((r1, r2), r3, rr_r)

    for r1, r2, r3 in rrr_temp:
        no_comm = ((r2, r1) in rr_r) and ((r3 not in rr_r[(r2, r1)]) or (len(rr_r[(r2, r1)]) > 1))
        comm = ((r2, r1, r3) in rrr_temp) and (not no_comm)

        if r1 in all_data.r_ht:
            for e1, e2 in all_data.r_ht[r1]:
                if (e2, r2) in all_data.hr_t:
                    for e3 in all_data.hr_t[(e2, r2)]:
                        if comm:
                            trips_comm.add((e1, r3, e3))
                        if no_comm:
                            trips_non_comm.add((e1, r3, e3))
    print("trips OK")
    pred(mers, comm_com, all_data, trips_comm, trips_non_comm)



def pred(mers, comm_com, all_data,trips_comm, trips_non_comm):
    for mer in mers:
        for model_name in ['FB237-600-full-0.5-model',
                           'FB237-600-base-model',
                           # 'FB237-600-sca-model',
                           # 'FB237-600-linear-model'
                           ]:
            patterns_mer = 0
            nums = 0
            with open('../%s_per_trips.json' % model_name, 'r', encoding='utf-8') as f:
                all_res = json.load(f)
            if comm_com:
                name = "comm_com"
            else:
                name = "non_comm_com"
            with open('./data/%s.txt' % name, 'w', encoding='utf-8') as f:
                for i, hrt in enumerate(all_data.data['test']):
                    h, r, t = hrt
                    if (comm_com and hrt in trips_comm and hrt not in trips_non_comm) or (not comm_com and hrt in trips_non_comm):
                        patterns_mer += all_res[mer][i]
                        nums += 1
                        f.write(h + " " + r + " " + t + "\n")

            print('model_name: %s' % model_name,
                  '三元组数量：', nums,
                  '指标 {} : {:.3f} '.format(mer, patterns_mer / (nums + 1e-9)))


def is_non_comm(r1, r2, r3, all_data):
    if r2 in all_data.r_ht:
        for e1, e2 in all_data.r_ht[r2]:
            if (e2, r1) in all_data.hr_t:
                for e3 in all_data.hr_t[(e2, r1)]:
                    if (e1, e3) in all_data.ht_r and all_data.ht_r[(e1, e3)] != r3:
                        return True
    return False


if __name__ == "__main__":
    # for top_num_value in [1, 2, 3, 4, 5, 7, 10, 50, 100, 200]:
    for top_num_value in [1000, 2000, 3000]:
        for comm_com in [True, False]:
            main_composition_ana(mers=['MRR', 'hits@1', 'hits@3', 'hits@10', 'MR'],
                                 comm_com=comm_com, top_num_value=top_num_value)


'''
可交换组合模式
43
data OK, rrr_temp len 868
trips OK
model_name: FB237-600-full-0.5-model 三元组数量： 6126 指标 MRR : 0.378 
model_name: FB237-600-base-model 三元组数量： 6126 指标 MRR : 0.367 
model_name: FB237-600-full-0.5-model 三元组数量： 6126 指标 hits@1 : 0.293 
model_name: FB237-600-base-model 三元组数量： 6126 指标 hits@1 : 0.279 
model_name: FB237-600-full-0.5-model 三元组数量： 6126 指标 hits@3 : 0.407 
model_name: FB237-600-base-model 三元组数量： 6126 指标 hits@3 : 0.401 
model_name: FB237-600-full-0.5-model 三元组数量： 6126 指标 hits@10 : 0.549 
model_name: FB237-600-base-model 三元组数量： 6126 指标 hits@10 : 0.544 
model_name: FB237-600-full-0.5-model 三元组数量： 6126 指标 MR : 151.680 
model_name: FB237-600-base-model 三元组数量： 6126 指标 MR : 161.980

43
data OK, rrr_temp len 868
trips OK
model_name: FB237-600-full-0.5-model 三元组数量： 279 指标 MRR : 0.471 
model_name: FB237-600-base-model 三元组数量： 279 指标 MRR : 0.400 
model_name: FB237-600-full-0.5-model 三元组数量： 279 指标 hits@1 : 0.387 
model_name: FB237-600-base-model 三元组数量： 279 指标 hits@1 : 0.317 
model_name: FB237-600-full-0.5-model 三元组数量： 279 指标 hits@3 : 0.493 
model_name: FB237-600-base-model 三元组数量： 279 指标 hits@3 : 0.412 
model_name: FB237-600-full-0.5-model 三元组数量： 279 指标 hits@10 : 0.634 
model_name: FB237-600-base-model 三元组数量： 279 指标 hits@10 : 0.575 
model_name: FB237-600-full-0.5-model 三元组数量： 279 指标 MR : 59.950 
model_name: FB237-600-base-model 三元组数量： 279 指标 MR : 117.394 
'''