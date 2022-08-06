import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# print('WN18RR')
# model_paths =[
# 	'/__data_root__/ZhouJie/MyKGE_2/logs/10_07/WN18RR/RotE_add_v10_05_28_45/model.pt',
# 	'../logs/04_14/WN18RR/AffE_17_04_26/model.pt',
# 	'../logs/04_15/WN18RR/AffE_15_39_31/model.pt',
# 	'../logs/04_15/WN18RR/AffE_07_52_20/model.pt',
# 	'../logs/04_14/WN18RR/AffE_23_58_21/model.pt'
# ]
# lbda = [0.01, 0, 1, 0.1, 0.001]
# mrr = [0.501, 0.483, 0.501, 0.502, 0.502]
#
# data = {'value': [], 'labda': []}
# for model_path, lb in zip(model_paths, lbda):
# 	model_para = torch.load(model_path)
# 	s = model_para['rel_mul.weight'].numpy().reshape(-1)
# 	print(lb, np.mean(s), np.var(s))


print('FB15k-237')
model_paths =[
	'../logs/04_30/FB237/AffE_02_16_22/model.pt',
	'../logs/04_17/FB237/AffE_10_48_22/model.pt',
	'../logs/04_16/FB237/AffE_23_03_19/model.pt',
	'../logs/04_16/FB237/AffE_11_16_02/model.pt',
	'../logs/04_15/FB237/AffE_23_27_30/model.pt',
	'../logs/04_30/FB237/AffE_08_23_00/model.pt',
	'../logs/05_02/FB237/AffE_04_14_55/model.pt',
	'../logs/05_02/FB237/AffE_14_43_06/model.pt'
]

lbda = [0, 1, 0.1, 0.01, 0.001, 0.001, 0.001, 0.0001]
mrr = [0.356, 0.353, 0.353, 0.353, 0.354, 0.358, 0.356,0.359]

data = {'value': [], 'labda': []}
for model_path, lb in zip(model_paths, lbda):
	model_para = torch.load(model_path)
	s = model_para['rel_mul.weight'].numpy().reshape(-1)
	print(lb, np.mean(s), np.var(s), np.max(s), np.min(s))