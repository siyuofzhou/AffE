import torch
import torch.optim
import numpy as np
import logging
import os
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params
import re
import random
import copy

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_model_from_pre2(model, model_path):
    pre_train_model = torch.load(model_path)
    state_dict = {}
    print(pre_train_model.keys())
    print(model.state_dict().keys())
    for k, v in model.state_dict().items():
        if k in pre_train_model:
            if 'rel' in k:
                c = v.cpu().numpy()
                b, d = pre_train_model[k].shape
                c[:b,:] = pre_train_model[k].numpy()
                state_dict[k] = torch.from_numpy(c)
            else:
                state_dict[k] = pre_train_model[k]
        #elif k in dense_model:
        #    if 'fc' in k:
        #        state_dict[k] = dense_model[k]
        #    else:
        #        state_dict[k] = v
        #    print('init by denseE: ', k)
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict)

def init_model_from_pre(model, model_path, reg):
    pre_train_model = torch.load(model_path)
    state_dict = {}
    print(pre_train_model.keys())
    print(model.state_dict().keys())
    # print(dense_model.keys())
    for k, v in model.state_dict().items():
        if k in pre_train_model and re.match(reg, k) is not None:
            print('pre model', k)
            state_dict[k] = pre_train_model[k]
        #elif k in dense_model:
        #    if 'fc' in k:
        #        state_dict[k] = dense_model[k]
        #    else:
        #        state_dict[k] = v
        #    print('init by denseE: ', k)
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict)

def init_logging(args , log_type = 'train'):
    save_dir = get_savedir(args.model.modelname, args.train.dataset)

    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, log_type+".log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))
    return save_dir


def init_logger(args, log_type = 'train'):
    save_dir = get_savedir(args.model.modelname, args.train.dataset)
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s')

    logger = logging.getLogger(str(args.config_id))
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(os.path.join(save_dir, log_type+".log"))
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    logger.info("Saving logs in: {}".format(save_dir))
    return save_dir


def get_type_datas(dataset, split, args):
    examples = dataset.get_examples(split)
    # 获得节点周围边列表和权重
    '''if args.rel_agg.use:
        topk_node_rels, topk_weight = dataset.get_node_rels("train", topk=args.rel_agg.topK)
        examples_head_rels = topk_node_rels[examples[:, 0]]
        head_topk_weight = topk_weight[examples[:, 0]]
        examples_tail_rels = topk_node_rels[examples[:, 2]]
        tail_topk_weight = topk_weight[examples[:, 2]]
        return torch.cat([examples, examples_head_rels, head_topk_weight, examples_tail_rels,tail_topk_weight], dim=-1)
    else:'''
    return examples

def get_datas(args, splits = None):
    if splits is None:
        splits = ['train','test','valid']
    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.train.dataset)
    dataset = KGDataset(dataset_path, args.train.debug, splits=copy.deepcopy(splits))
    args.model.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    res = [dataset]
    for split in splits:
        examples = get_type_datas(dataset, split, args)
        res.append(examples)
    #filters = dataset.get_filters()
    return res


def train(args, save_dir, pre_model_train=None):
    setup_seed(args.test)
    logger = logging.getLogger(str(args.config_id))
    logger.propagate = False
    dataset, train_examples, valid_examples, test_examples = get_datas(args,splits=['train','valid','test'])
    
    # if args.rel_agg.use == 'rel':
    #     topk_node_rels, topk_weight = dataset.get_node_rels("train", topk=args.rel_agg.topK)
    # elif args.rel_agg.use == 'rel_tail':
    #     topk_node_rels, topk_node_tails, topk_node_mask = dataset.get_node_r_t("train", topk=args.rel_agg.topK)

    # 对称关系id
    # sym_rel_ids = dataset.sym_rel_ids
    filters = dataset.get_filters()

    # save config
    # with open(os.path.join(save_dir, "config.json"), "w") as fjson:
    #     json.dump(vars(args), fjson)
    ## logging.info('\t Config_id: '+str(args.config_id))
    logger.info('Config_id: '+str(args.config_id))

    # create model
    '''
    if args.rel_agg.use == 'rel':
        args.model.topK = args.rel_agg.topK
        args.model.topK_alpha = args.rel_agg.alpha
        model = getattr(models, args.model.modelname)(args.model, topk_node_rels, topk_weight)
    elif args.rel_agg.use == 'rel_tail':
        args.model.topK = args.rel_agg.topK
        args.model.topK_alpha = args.rel_agg.alpha
        model = getattr(models, args.model.modelname)(args.model, topk_node_rels, topk_node_tails, topk_node_mask)
    else:
        model = getattr(models, args.model.modelname)(args.model)
    '''
    model = getattr(models, args.model.modelname)(args.model)
    total = count_params(model)
    ## logging.info("Total number of parameters {}".format(total))
    logger.info("Total number of parameters {}".format(total))
    # torch.cuda.set_device(1)
    device = "cuda"
    model.to(device)
    # 对称关系id
    # model.set_sym_rel_ids(sym_rel_ids)
    if args.load.use_pre_model:
        init_model_from_pre2(model,'%s/model.pt'%args.load.pre_model)
    # get optimizer
    regularizer = getattr(regularizers, args.regularizer.reg_name)(args.regularizer)
    optim_method = getattr(torch.optim, args.train.optimizer)(model.parameters(), lr=args.train.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.train.batch_size, args.train.neg_sample_size,
                            bool(args.train.double_neg))
    
    counter = 0
    best_mrr = None
    best_epoch = None
    ## logging.info("\t Start training")
    logger.info("\t Start training")
    for step in range(args.train.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        ## logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))
        logger.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        ## logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))
        logger.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.train.valid == 0 and (step+1) >= args.train.vail_start_epoch:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            ## logging.info(format_metrics(valid_metrics, split="valid"))
            logger.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                ## logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                logger.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model.cuda()
            else:
                counter += 1
                if counter == args.train.patience:
                    ## logging.info("\t Early stopping")
                    logger.info("\t Early stopping")
                    break
                elif counter == args.train.patience // 2:
                    pass
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()

    ## logging.info("\t Optimization finished")
    logger.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        ## logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        logger.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    if pre_model_train is not None:
        torch.save(model.cpu().state_dict(), pre_model_train)
    model.cuda()
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
    ## logging.info(format_metrics(valid_metrics, split="valid"))
    logger.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    ## logging.info(format_metrics(test_metrics, split="test"))
    logger.info(format_metrics(test_metrics, split="test"))

    return valid_metrics, test_metrics, os.path.join(save_dir, "model.pt")


def test(args, save_dir):
    logger = logging.getLogger(str(args.config_id))
    logger.propagate = False
    res = get_datas(args, splits=args.load.test_dataset)
    dataset = res[0]
    filters = dataset.get_filters()
    
    logger.info('Config_id: ' + str(args.config_id))
    
    # create model
    model = getattr(models, args.model.modelname)(args.model)
    total = count_params(model)
    ## logging.info("Total number of parameters {}".format(total))
    logger.info("Total number of parameters {}".format(total))
    torch.cuda.set_device(1)
    device = "cuda:1"
    model.to(device)
    
    # 对称关系id
    
    init_model_from_pre(model, './pre_model/%s.pt' % args.load.pre_model, args.load.key_reg)
    # get optimizer
    regularizer = getattr(regularizers, args.regularizer.reg_name)(args.regularizer)
    optim_method = getattr(torch.optim, args.train.optimizer)(model.parameters(), lr=args.train.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.train.batch_size, args.train.neg_sample_size,
                            bool(args.train.double_neg))
    model.eval()
    all_metrics = {}
    print(len(res), args.load.test_dataset)
    for ds, name in zip(res[1:], args.load.test_dataset):
        metrics = avg_both(*model.compute_metrics(ds, filters))
        logger.info(format_metrics(metrics, split=name))
        all_metrics[name] = metrics
    print('OK')
    return all_metrics, os.path.join(save_dir, "model.pt")

if __name__ == '__main__':
    print(re.match('(entity|rel|bh|bt).*','re.weight'))