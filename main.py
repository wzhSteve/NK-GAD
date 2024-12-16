import logging
import numpy as np
from tqdm import tqdm
import torch

import sys
# sys.path.append('pygod')
import importlib
import random

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
from pygod.utils import load_data
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops
from torch_geometric.seed import seed_everything

from pygod.metric import *
from nkad import NKAD

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils import *
import logging


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    num_hidden = args.num_hidden

    node_encoder_num_layers = args.node_encoder_num_layers
    attr_decoder_num_layers= args.attr_decoder_num_layers
    struct_decoder_num_layers= args.struct_decoder_num_layers

    attr_encoder_name = args.attr_encoder
    attr_decoder_name = args.attr_decoder
    struct_decoder_name= args.struct_decoder

    weight_decay=args.weight_decay
    optim_type = args.optimizer 

    lr = args.lr

    model_name=args.model_name
    aggr_f=args.aggr_f
    alpha= args.alpha
    dropout=args.dropout
    loss_f=args.loss_f
    loss_weight_f=args.loss_weight_f
    T_f = args.T_f

    # graph preprocess
    graph = load_data(dataset_name, current_dir+"/data")
    graph.edge_index = add_remaining_self_loops(graph.edge_index)[0]
    graph.s = to_dense_adj(graph.edge_index)[0]
    num_features = graph.x.size()[1]
    # num_classes=4

    args.num_features = num_features

    auc_score_list = []
    ap_score_list = []
    rec_score_list = []

    auc_score_list_c = []
    ap_score_list_c = []
    rec_score_list_c = []

    auc_score_list_s = []
    ap_score_list_s = []
    rec_score_list_s = []

    for i, seed in enumerate(seeds):
        logging.info(f"####### Run {i} for seed {seed}")
        # seed=int(seed)
        set_random_seed(seed)
        seed_everything(seed)

        logger = None
        model= eval(model_name)(epoch=max_epoch,
                                aggr=aggr_f, 
                                hid_dim=num_hidden,
                                gpu=device,
                                alpha=alpha,
                                dropout=dropout,
                                lr=lr,loss_name=loss_f,
                                loss_weight=loss_weight_f,
                                weight_decay=weight_decay,
                                T=T_f,
                                attr_encoder_name=attr_encoder_name,
                                attr_decoder_name=attr_decoder_name,
                                struct_decoder_name=struct_decoder_name,
                                node_encoder_num_layers=node_encoder_num_layers,
                                attr_decoder_num_layers=attr_decoder_num_layers,
                                struct_decoder_num_layers=struct_decoder_num_layers,
                                highpass_layer_num=args.highpass_layer_num,
                                neighbor_rec_loss_coe=args.neighbor_rec_loss_coe,
                                high_coe=args.high_coe,
                                center_rec_coef=args.center_rec_coef)

        model.fit(graph)
        # labels = model.predict(graph)

        outlier_scores = model.decision_function(graph)
        edge_outlier_scores = model.decision_struct_function(graph)

        y = graph.y.bool().cpu()

        outlier_scores = outlier_scores #+ decision_score

        auc_score = eval_roc_auc(y, outlier_scores)
        ap_score = eval_average_precision(y, outlier_scores)
        rec_score = eval_recall_at_k(y, outlier_scores, torch.sum(y))
        auc_score_c, ap_score_c, rec_score_c = None, None, None
        auc_score_s, ap_score_s, rec_score_s = None, None, None
        if "inj_" in dataset_name:
            y_c = graph.y.cpu() >> 0 & 1 # contextual outliers
            y_s = graph.y.cpu() >> 1 & 1 # structural outliers
            auc_score_c = eval_roc_auc(y_c, outlier_scores)
            ap_score_c = eval_average_precision(y_c, outlier_scores)
            rec_score_c = eval_recall_at_k(y_c, outlier_scores, sum(y_c))

            auc_score_s = eval_roc_auc(y_s, outlier_scores)
            ap_score_s = eval_average_precision(y_s, outlier_scores)
            rec_score_s = eval_recall_at_k(y_s, outlier_scores, sum(y_s))
        logging.info(f'auc_score: {auc_score:.4f}')


        auc_score_list.append(auc_score)
        ap_score_list.append(ap_score)
        rec_score_list.append(rec_score)

        if "inj_" in dataset_name:
            auc_score_list_c.append(auc_score_c)
            ap_score_list_c.append(ap_score_c)
            rec_score_list_c.append(rec_score_c)
            auc_score_list_s.append(auc_score_s)
            ap_score_list_s.append(ap_score_s)
            rec_score_list_s.append(rec_score_s)

        if logger is not None:
            logger.finish()

    auc = torch.tensor(auc_score_list)
    ap = torch.tensor(ap_score_list)
    rec = torch.tensor(rec_score_list)
    if "inj_" in dataset_name:
        auc_c = torch.tensor(auc_score_list_c)
        ap_c = torch.tensor(ap_score_list_c)
        rec_c = torch.tensor(rec_score_list_c)
        auc_s = torch.tensor(auc_score_list_s)
        ap_s = torch.tensor(ap_score_list_s)
        rec_s = torch.tensor(rec_score_list_s)

    logging.info(dataset_name + " " + 
          "AUC: {:.4f}±{:.4f} ({:.4f})\t"
          "AP: {:.4f}±{:.4f} ({:.4f})\t"
          "Recall: {:.4f}±{:.4f} ({:.4f})".format(torch.mean(auc),
                                                  torch.std(auc),
                                                  torch.max(auc),
                                                  torch.mean(ap),
                                                  torch.std(ap),
                                                  torch.max(ap),
                                                  torch.mean(rec),
                                                  torch.std(rec),
                                                  torch.max(rec)))
    if "inj_" in args.dataset:
        logging.info(dataset_name + " " + " contextual: " +
            "AUC: {:.4f}±{:.4f} ({:.4f})\t"
            "AP: {:.4f}±{:.4f} ({:.4f})\t"
            "Recall: {:.4f}±{:.4f} ({:.4f})".format(torch.mean(auc_c),
                                                    torch.std(auc_c),
                                                    torch.max(auc_c),
                                                    torch.mean(ap_c),
                                                    torch.std(ap_c),
                                                    torch.max(ap_c),
                                                    torch.mean(rec_c),
                                                    torch.std(rec_c),
                                                    torch.max(rec_c)))
        logging.info(dataset_name + " " + " structural: " +
            "AUC: {:.4f}±{:.4f} ({:.4f})\t"
            "AP: {:.4f}±{:.4f} ({:.4f})\t"
            "Recall: {:.4f}±{:.4f} ({:.4f})".format(torch.mean(auc_s),
                                                    torch.std(auc_s),
                                                    torch.max(auc_s),
                                                    torch.mean(ap_s),
                                                    torch.std(ap_s),
                                                    torch.max(ap_s),
                                                    torch.mean(rec_s),
                                                    torch.std(rec_s),
                                                    torch.max(rec_s)))



# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    args.seeds = [i for i in range(10)]
    args = load_best_configs(args, current_dir + "/config_ada-gad.yml")

    if args.alpha=='None':
        args.alpha = None

    # print(args)
    
    main(args)
    

