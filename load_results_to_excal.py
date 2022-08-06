import numpy as np
import pandas as pd
import os
import json
import yaml

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', None)
float_format = lambda x:'%.8f'%x


def to_float(x):
	try:
		return float(x)
	except:
		return x

def splicing_keys(record):
	data = {}
	def dfs(pre_key, kv):
		if isinstance(kv,dict):
			for k, v in kv.items():
				dfs(pre_key+'<:>'+k, v)
		else:
			data[pre_key] = to_float(kv)
	dfs('', record)
	return data


def load_results(result_dir):
	files = os.listdir(result_dir)
	for file_name in files:
		with open(os.path.join(result_dir, file_name),'r',encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if len(line) == 0:
					continue
				columns = json.loads(line)
				columns = splicing_keys(columns)
				yield columns
			# data = pd.DataFrame(np.array(all_values), columns=[first_columns,second_columns])


def load_yaml(config_dir, config_id):
	with open(os.path.join(config_dir,str(int(config_id))+'.yaml'),'r', encoding='utf-8') as f:
		file = yaml.safe_load(f)
		data = splicing_keys(file)
	return data

# load_results('./result')
# load_yaml(1)

def load_data(result_dirs, config_dirs):
	keys = set()
	data = []
	for result_dir, config_dir in zip(result_dirs, config_dirs):
		for columns in load_results(result_dir):
			config = load_yaml(config_dir, columns['<:>config_id'])
			data.append([columns, config])
			for key in columns.keys():
				keys.add(key)
			for key in config.keys():
				keys.add(key)
	
	keys = sorted(list(keys))
	print(keys)
	first_columns = []
	second_columns = []
	all_values = []
	for key in keys:
		ks = key.split('<:>')
		if len(ks) == 2:
			first_columns.append(key.split('<:>')[1])
			second_columns.append(key.split('<:>')[1])
		else:
			first_columns.append(key.split('<:>')[1])
			second_columns.append(key.split('<:>')[2])
			
	
	for columns, config in data:
		values = []
		for key in keys:
			if key in columns:
				values.append(columns[key])
			elif key in config:
				values.append(config[key])
			else:
				values.append('')
		all_values.append(values)
	# print(np.array(all_values).shape)
	pd_data = pd.DataFrame(np.array(all_values), columns=[first_columns, second_columns])
	# print(set(first_columns))
	pd_data = pd_data[['config_id','model_path','load','rel_agg','regularizer','model','train','vaild','test']]
	# print(pd_data)
	pd_data.to_excel('result_data_4.xls',sheet_name='test')
		

# load_data(['/__data_root__/ZhouJie/MyKGE/result','/__data_root__/ZhouJie/MyKGE_2/result'],
# 		  ['/__data_root__/ZhouJie/MyKGE/config_history','/__data_root__/ZhouJie/MyKGE_2/config_history'])
load_data(['/__data_root__/ZhouJie/MyKGE/YAGO3-10_logs/result','/__data_root__/ZhouJie/MyKGE/result','/__data_root__/ZhouJie/MyKGE_2/result'],
		  ['/__data_root__/ZhouJie/MyKGE/YAGO3-10_logs/config_history','/__data_root__/ZhouJie/MyKGE/config_history','/__data_root__/ZhouJie/MyKGE_2/config_history'])