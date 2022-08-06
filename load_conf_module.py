import yaml
import os
import copy
import hashlib
from utils.train import Dict

# 根据路径名称载入配置
def load_config_by_path(path):
    if os.path.exists(path):
        with open(path,'r') as f:
            _config = yaml.safe_load(f)
        return _config
    print('No pre model config')
    return None

def load_config(enum_path=None):
    '''
    读取默认配置、枚举配置、过滤配置、网格搜索配置
    :return: 默认配置、枚举配置、过滤配置、网格搜索配置
    '''
    if os.path.exists('./config/default_configuration_file.yaml'):
        with open('./config/default_configuration_file.yaml','r') as f:
            default_config = yaml.safe_load(f)
    
    if enum_path is None:
        enum_path = './config/enum_configuration_file.yaml'
    else:
        enum_path = os.path.join('./config',enum_path)
    
    if os.path.exists(enum_path):
        with open(enum_path,'r') as f:
            enum_config = yaml.safe_load(f)

    if os.path.exists('./config/filter_configuration_file.yaml'):
        with open('./config/filter_configuration_file.yaml','r') as f:
            filter_config = yaml.safe_load(f)

    return default_config,enum_config,filter_config

'''
_dfs_config 讲配置中的键拼接起来形成哈希表的key.
_config的内容如下
train:
    - a : v1
    - c : [v1,v2]
    - a : v3
    - b :
        d : v4
经过_dfs_config后得到的哈希表为：
{':train:0:a'     :  [v3]}
{':train:1:c'     :  [v1,v2]}
{':train:3:b:d'   :  [v4]}
添加index, 防止同名覆盖
'''
def _dfs_config(_config,pre_key,params_map):
    if isinstance(_config,list):
        state = _check_list_vaild(_config)
        if state == -1:
            return False
        elif state == 1:
            params_map[pre_key] = _config
        else:
            for i,u in enumerate(_config):
                key_str = pre_key
                if _dfs_config(u, key_str+":%d"%i, params_map) is False:
                    return False
    elif isinstance(_config,dict):
        for k,v in _config.items():
            if _dfs_config(v,pre_key+":"+str(k),params_map) is False:
                return False
    else:
        params_map[pre_key] = [_config]
    return True

def _check_list_vaild(node):
    state = -1
    for u in node:
        if isinstance(u,list) or isinstance(u, dict):
            if state == -1 or state == 2:
                state = 2
            else:
                return -1
        else:
            if state == -1 or state == 1:
                state = 1
            else:
                return -1
    return state


'''
网格搜索_params_list中的val的可能值，替换params中的值，返回一个迭代器，
迭代需要遍历的参数。
'''
def _dfs_params(params, _params_list):
    list_index = [0 for _ in range(len(_params_list))]
    list_limit = [len(vals) for k,vals in _params_list]
    while True:
        _params = copy.deepcopy(params)
        for i,index in enumerate(list_index):
            k,vals = _params_list[i]
            _params[k] = [vals[index]]
        for k,v in _params.items():
            _params[k] = v[0]
        yield _params
        if next_list_index(list_limit, list_index) is False:
            break


def next_list_index(list_limit, list_index):
    for i in range(len(list_index)-1,-1,-1):
        list_index[i] += 1
        if list_index[i] < list_limit[i]:
            return True
        else:
            list_index[i] = 0
    return False

'''
遍历配置文件中的case，对每个case中的配置进行网格搜索，返回一个迭代器，迭代器每次生成一个参数配置
'''
def traversal_configs(default_config, config):
    default_params = dict()
    if _dfs_config(default_config, "", default_params) is False:
        print('default config is not vaild!')
        return
    for i,_config in enumerate(config):
        print('case:')
        enum_params = dict()
        if _dfs_config(_config['case'], "", enum_params) is False:
            print('case:',i,'is not vaild!')
            continue
        #print(enum_params)
        for params in _dfs_params(copy.deepcopy(default_params), list(enum_params.items())):
            #print('params',params)
            yield params


'''
过滤配置文件，将过滤配置文件中的配置使用默认配置补全后，计算配置的哈希值
'''
def filter_config_op(default_config, filter_config):
    filter_table = dict()
    for params in traversal_configs(default_config, filter_config):
        hash_val, hash_val1, hash_val2 = dict_to_md5(params)
        filter_table[hash_val] = [hash_val1,hash_val2]
    return filter_table

def dict_to_md5(params_dict):
    params_str = dict_to_str(params_dict)
    n = len(params_str)
    hash_val = stringtomd5(params_str.encode('utf-8'))
    hash_val1 = stringtomd5(params_str[:n // 2].encode('utf-8'))
    hash_val2 = stringtomd5(params_str[n // 2:].encode('utf-8'))
    return hash_val, hash_val1, hash_val2

def stringtomd5(originstr):
    """将string转化为MD5"""
    signaturemd5 = hashlib.md5()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

'''
检查配置是否再过滤表中，如果在返回True，否则返回False
'''
def check_filter_table(hashs, filter_table):
    hash_val,hash_val1,hash_val2 = hashs
    if hash_val in filter_table:
        filter_hash_val1, filter_hash_val2 = filter_table[hash_val]
        if hash_val1 == filter_hash_val1 and hash_val2 == filter_hash_val2:
            return True
    return False


def dict_to_str(params_dict):
    params_list = list(params_dict.items())
    params_list = sorted(params_list)
    concat_str = '++'.join([str(k) for k,v in params_list if 'config_id' not in k]) \
                 + '-'.join([str(v) for k,v in params_list if 'config_id' not in k])
    return concat_str

'''
将配置表转成嵌套字典
'''
def params_to_config(_config,pre_key,params_map):
    _new_config = None
    if isinstance(_config, list):
        state = _check_list_vaild(_config)
        if state == -1 and state == 1:
            return None
        else:
            _new_config = []
            for i, u in enumerate(_config):
                ret = params_to_config(u, pre_key+":%d"%i, params_map)
                if ret is None:
                    return None
                _new_config.append(ret)
    elif isinstance(_config, dict):
        _new_config = {}
        for k, v in _config.items():
            ret = params_to_config(v, pre_key + ":" + str(k), params_map)
            if ret is None:
                return None
            _new_config[k] = ret
    else:
        return params_map[pre_key]
    return _new_config

def iter_config(save_config=True, enum_path=None):
    # 配置载入
    default_config, enum_config, filter_config = load_config(enum_path=enum_path)
    # print(default_config)
    # 过滤配置文件，将过滤配置文件中的配置使用默认配置补全后，计算配置的哈希值
    filter_table = filter_config_op(default_config, filter_config)

    if os.path.exists('config_history') is False:
        os.makedirs('config_history')
    # 获取历史配置的hash值
    config_index = load_history_config_hash(filter_table)
    
    # 枚举配置文件
    for params in traversal_configs(default_config, enum_config):
        hashs = dict_to_md5(params)
        if check_filter_table(hashs, filter_table):
            continue
        filter_table[hashs[0]] = hashs[1:]
        _config = copy.deepcopy(default_config)
        _new_config = params_to_config(_config, "", params)
        _new_config["config_id"] = config_index
        yield _new_config
        if save_config:
          with open('config_history/%d.yaml'%(config_index), 'w', encoding='utf-8') as f:
            yaml.dump(_new_config, f, allow_unicode=True)  # allow_unicode=True在数据中存在中文时，解决编码问题
        config_index += 1


def load_history_config_hash(filter_table):
    config_paths = os.listdir('config_history')
    config_index = len(config_paths) + 1
    for index,config_path in enumerate(config_paths):
        if os.path.exists('config_history/%d.yaml'%(index+1)):
            with open('config_history/%d.yaml'%(index+1),'r') as f:
                history_config = yaml.safe_load(f)
            params = dict()
            _dfs_config(history_config,"",params)
            for k,v in params.items():
                params[k] = v[0]
            hash = dict_to_md5(params)
            filter_table[hash[0]] = hash[1:]
    return config_index

if __name__ == "__main__":
    for config in iter_config(save_config=False):
        print(config)






