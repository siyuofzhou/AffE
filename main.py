from train import train,test, init_logging, init_logger
from load_conf_module import iter_config,load_config_by_path
from utils.train import Dict
import json
import os
import logging
import yaml

os.environ['LOG_DIR'] = "./logs"
os.environ['DATA_PATH'] = "./data"

def check_pre_model(args, save_dir):
    pre_model_path = './pre_model/%s.pt'%args.load.pre_model
    if os.path.exists(pre_model_path) is False:
        logging.info("pre train model: %s"%args.load.pre_model)
        pre_model_config = load_config_by_path('./pre_model_config/%s.yaml'%args.load.pre_model)
        pre_model_args = Dict(pre_model_config)
        train(pre_model_args, save_dir, pre_model_train = pre_model_path)


def save_result(args, valid_metrics, test_metrics, model_path):
    mertics = {
        'config_id': args.config_id,
        'test': {
            "MR": str(float(test_metrics['MR'])),
            "MRR": str(float(test_metrics['MRR'])),
            "hits@1": str(float(test_metrics['hits@[1,3,10]'][0])),
            "hits@3": str(float(test_metrics['hits@[1,3,10]'][1])),
            "hits@10": str(float(test_metrics['hits@[1,3,10]'][2])),
        },
        'vaild': {
            "MR": str(float(valid_metrics['MR'])),
            "MRR": str(float(valid_metrics['MRR'])),
            "hits@1": str(float(valid_metrics['hits@[1,3,10]'][0])),
            "hits@3": str(float(valid_metrics['hits@[1,3,10]'][1])),
            "hits@10": str(float(valid_metrics['hits@[1,3,10]'][2])),
        },
        'model_path': model_path
    }
    # 结果文件名构建
    name = str(args.model.modelname)+'_'+str(args.train.dataset)+'_'+str(args.model.rank)+'.log'

    with open('result/'+name, 'a', encoding='utf-8') as f:
        f.write('\n')
        json.dump(mertics, f)


def save_result2(args, all_metrics):
    new_all_mertics = {
        'config_id': args.config_id,
    }
    for k,mertics in all_metrics.items():
        new_all_mertics[k] = {
            "MR": str(float(mertics['MR'])),
            "MRR": str(float(mertics['MRR'])),
            "hits@1": str(float(mertics['hits@[1,3,10]'][0])),
            "hits@3": str(float(mertics['hits@[1,3,10]'][1])),
            "hits@10": str(float(mertics['hits@[1,3,10]'][2])),
        }
    # 结果文件名构建
    name = str(args.model.modelname)+'_'+str(args.train.dataset)+'_'+str(args.model.rank)+'.log'

    with open('result/'+name, 'a', encoding='utf-8') as f:
        f.write('\n')
        json.dump(new_all_mertics, f)

class DictToStr:
    def __init__(self):
        self.s = '\n'

    def dict_to_str(self, c, pre = 0):
        if isinstance(c,dict):
            self.s = self.s+'\n'
            for k,v in c.items():
                self.s = self.s + ' '*pre + str(k) + ':'
                self.dict_to_str(v, pre+2)
        else:
            self.s = self.s + " " + str(c) + '\n'

def run_train():
    save_dir = None
    # 从配置文件中枚举配置
    for config in iter_config(enum_path='FB15k-237_config.yaml'):
        args = Dict(config)
        # 初始化logging
        save_dir = init_logger(args)
        # 检查预训练模型是否存在，如果不存在则添加训练
        # if args.load.use_pre_model:
        #     check_pre_model(args, save_dir)
        
        logger = logging.getLogger(str(args.config_id))
        logger.propagate = False
        con = DictToStr()
        con.dict_to_str(config)
        logger.info(con.s)
        
        valid_metrics, test_metrics, model_path = train(args, save_dir)
        save_result(args, valid_metrics, test_metrics, model_path)
        

def run_test():
    # 从配置文件中枚举配置
    if os.path.exists('./config/predict_config.yaml'):
        with open('./config/predict_config.yaml','r') as f:
            predict_config = yaml.safe_load(f)
            
    predict_args = Dict(predict_config)
    # 初始化logging
    save_dir = init_logger(predict_args, log_type='test')
    # 检查预训练模型是否存在，如果不存在则添加训练
    check_pre_model(predict_args, save_dir)
    all_metrics, model_path = test(predict_args, save_dir)
    save_result2(predict_args, all_metrics)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_train()
    # run_test()
