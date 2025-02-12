# @Organization  : DBPR
# @Author        : DBPR
# @Time          : 2023/9/6 10:01 PM
# @Function      : Train and Test Model

import time
import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import copy
import datetime
import random
from model import *
from utils import *
#from utils_Package.weight_perturbation import WPOptim
from lw2w import WeightNetwork, LossWeightNetwork, FeatureMatching, inner_objective, inner_objective1,outer_objective, validate
from l2t_ww.train.meta_optimizers import MetaSGD
from l2t_ww.check_model import check_model
from N_data_dataloaders import recorded_llm_multicenters_dataloader
from focal_loss import FocalLoss
from torch.nn.parallel import DataParallel
from torchsummary import summary


def get_args():
    #framework hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="UNet", help='neural network used in training')
    parser.add_argument('--dataset', type=str, default="nuclei", help='dataset_train used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid',
                        help='the data partitioning strategy only use for cifar100,cifar10,tinyimagenet,lidc:homo,iid,noniid-labeldir,noniid')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='input batch size for training (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--alpha_PI', type=float, default=0.5, help='prediction Imitation alpha(default: 0.5)')
    parser.add_argument('--temp', type=float, default=500, help='the temperature for FI')
    parser.add_argument('--TV', type=float, default=4, help='the threshold value for DDBL')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--alg', type=str, default='llm_fedlwt',
                        help='communication strategy: llm_fedlwt')
    parser.add_argument('--llm', type=str, default="dinov2",
                        help=': dinov2')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication round')
    parser.add_argument('--init_seed', type=int, default=4000, help="Random seed")
    parser.add_argument('--input-shape', type=int, default=256)
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default='./data/nuclei', help="Data directory")
    parser.add_argument('--reg', type=float, default=0.0001, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default='./model_save/dinov2_2024-6-25_llm_fedlwt_UNet_nuclei/logs_auc_ratio/',
                        help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default='./model_save/dinov2_2024-6-25_llm_fedlwt_UNet_nuclei/models_auc_ratio/',
                        help='Model directory path')
    parser.add_argument('--beta_distribution', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='deprecated')
    parser.add_argument('--out_dim', type=int, default=256, help='deprecated')
    parser.add_argument('--temperature', type=float, default=0.5, help='deprecated')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='deprecated')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='deprecated')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None,help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='deprecated')
    parser.add_argument('--normal_model', type=int, default=0, help='deprecated')
    parser.add_argument('--loss_func', type=str, default='DiceLoss', help="crossentropy、focalloss、DiceLoss")
    parser.add_argument('--focalloss_alpha', type=str, default=None)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_model_interval', type=int, default=1)
    parser.add_argument('--use_project_head', type=int, default=0, help='deprecated')
    parser.add_argument('--server_momentum', type=float, default=0, help='deprecated')
    parser.add_argument('--global_pre_join_with_new_global', type=float, default=3,help='deprecated')
    parser.add_argument('--alpha', type=float, default=0.05, help='deprecated')
    parser.add_argument('--open_perturbe', type=int, default=False, help='deprecated')
    ##############HGKT hyper-parameters################
    parser.add_argument('--num-classes', type=int, default=2, help='the number of category')
    parser.add_argument('--source-model', default="dinov2", type=str)
    parser.add_argument('--source-domain', default='imagenet', type=str,help='deprecated')
    parser.add_argument('--source-path', type=str, default=None)
    parser.add_argument('--target-model', default='UNet', type=str)
    parser.add_argument('--target_model_pre_weights_path', default=None, type=str)
    parser.add_argument('--weight-path', type=str, default=None,help='deprecated')
    parser.add_argument('--wnet-path', type=str, default=None)
    parser.add_argument('--open_lw2w', type=int, default=False, help='model_pre_matching')
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--schedule', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pairs', type=list, default=[(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4)])
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Initial learning rate for meta networks')
    parser.add_argument('--meta-wd', type=float, default=1e-3)
    parser.add_argument('--loss-weight', action='store_true', default=True)
    parser.add_argument('--loss-weight-type', type=str, default='relu6')
    parser.add_argument('--loss-weight-init', type=float, default=1.0)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--source-optimizer', type=str, default='sgd')
    parser.add_argument('--experiment', default='logs', help='Where to store models')
    parser.add_argument('--target-mhsa', type=bool, default=False, help="utilize the mhsa,deprecated")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--imbalance',  type=bool, default=True, help='do not truncate train data to same length')
    args = parser.parse_args()
    return args

def init_pairs(opt):
    return opt.pairs

def init_meta_model(opt, pairs, server_model, local_nets):
    """
    Initialize independent meta-network parameters for each user endpoint, used for lightweight vision foundation models.
    :opt: hyper-parameters
    :pairs: predefined model matching layers between the vision foundation model and the target model.
    :server_model: deprecated
    :returns: meta-network parameters
    """
    local_models_name = opt.sites.copy()
    server_model_name=[opt.llm]#dinov2
    wnets = dict()
    lwnets = dict()
    wlw_weight_params = dict()
    target_params_dict = dict()
    target_branch_dict = dict()
    for net_i in local_models_name:
        wnet = WeightNetwork(opt.source_model, pairs)
        weight_params = list(wnet.parameters())
        if opt.loss_weight:
            lwnet = LossWeightNetwork(opt.source_model, pairs, opt.loss_weight_type, opt.loss_weight_init)
            weight_params = weight_params + list(lwnet.parameters())
        if opt.wnet_path is not None:
            ckpt = torch.load(opt.wnet_path)
            wnet.load_state_dict(ckpt['w'])
            if opt.loss_weight:
                lwnet.load_state_dict(ckpt['lw'])

        target_branch = FeatureMatching(opt.source_model,opt.target_model,pairs)
        local_target_params = list(local_nets[net_i].parameters()) + copy.deepcopy(list(target_branch.parameters()))
        wnets["%sto%s" % (server_model_name[0], net_i)] = copy.deepcopy(wnet)
        lwnets["%sto%s" % (server_model_name[0], net_i)] = copy.deepcopy(lwnet)
        wlw_weight_params["%sto%s" % (server_model_name[0], net_i)] = copy.deepcopy(weight_params)
        target_params_dict["%sto%s" % (server_model_name[0], net_i)] = local_target_params
        target_branch_dict["%sto%s" % (server_model_name[0], net_i)] = copy.deepcopy(target_branch)
    return wnets, lwnets, wlw_weight_params, target_params_dict, target_branch_dict  # ,source_optimizer

def optimizer_init(opt, wlw_weight_params, target_model_dict, target_params_dict, target_branch_dict):
    """
    Initialize optimizers for each client to update local models and meta-network parameters.
    :opt: hyper-parameters
    :wlw_weight_params: meta-network model parameters for each client
    :target_model_dict: local models for each client
    :target_params_dict: local model parameters for each client
    :target_branch_dict: branch networks for each client used to compute the feature transfer loss function
    :returns: optimizers
    """
    source_optimizers = dict()  
    target_optimizers = dict()
    for i, (k, weight_params) in enumerate(wlw_weight_params.items()):
        if opt.source_optimizer == 'sgd':
            source_optimizer = optim.SGD(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd, momentum=opt.momentum,
                                         nesterov=opt.nesterov)
        elif opt.source_optimizer == 'adam':
            source_optimizer = optim.Adam(weight_params, lr=opt.meta_lr, weight_decay=opt.meta_wd)
        source_optimizers[k] = source_optimizer
        if opt.meta_lr == 0:
            target_optimizer = optim.SGD(target_params_dict[k], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)
        else:
            target_optimizer = MetaSGD(target_params_dict[k],
                                       [target_model_dict[k.split("to")[-1]], target_branch_dict[k]],
                                       lr=opt.lr,
                                       momentum=opt.momentum,
                                       weight_decay=opt.wd, rollback=True, cpu=opt.T > 2)
        target_optimizers[k] = target_optimizer
    return source_optimizers, target_optimizers

def init_nets(args, device='cpu', server=False):
    """
    Initialize local models for each client and shared model parameters.
    :opt: hyper-parameters
    :returns: local models for each client
    """
    if args.alg == "fedlwt":
        local_models = args.sites.copy()
        server_model = [args.sites[args.global_center_idx]]
        local_models.pop(args.global_center_idx)
    else:
        local_models = args.sites.copy()
        server_model = ["server"]
    #script_directory = os.path.dirname(os.path.abspath(__file__))
    if args.target_model_pre_weights_path is not None:
        checkpoint = torch.load(args.target_model_pre_weights_path, map_location=device)
    #checkpoint = torch.load(args.target_model_pre_weights_path, map_location=device)
    
    if server:
        nets = {net_i: None for net_i in server_model}
        for net_i in server_model:
            net = check_model(args).to(device)
            ###finetune imagenet
            #print(net)
            new_params = net.state_dict().copy()
            #for name, param in new_params.items():
                #print(name)
            if args.target_model_pre_weights_path is not None:
                new_params = net.state_dict().copy()
                for name, param in new_params.items():
                    # print(name)
                    if name in checkpoint and param.size() == checkpoint[name].size():
                        new_params[name].copy_(checkpoint[name])
                        # print('copy {}'.format(name))
                net.load_state_dict(new_params)
            nets[net_i] = net
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[net_i].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)
        return nets, model_meta_data, layer_type
    else:
        nets = {net_i: None for net_i in local_models}
        for net_i in local_models:
            net = check_model(args).to(device)
            ###finetune imagenet
            if args.target_model_pre_weights_path is not None:
                new_params = net.state_dict().copy()
                #print(net)
                for name, param in new_params.items():
                    #print(name)
                    if name in checkpoint and param.size() == checkpoint[name].size():
                        new_params[name].copy_(checkpoint[name])
                        # print('copy {}'.format(name))
                net.load_state_dict(new_params)
            nets[net_i] = net
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[net_i].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

        return nets, model_meta_data, layer_type

def llm_init(llm_name):
    """
    Initialize the parameters for the vision foundation model.
    :llm_name: name of the vision foundation model
    :returns: vision foundation model
    """
    if llm_name.split("_")[0]=="dinov2":
        llm_model = torch.hub.load('', '%s_vitb14'%llm_name, source='local')
    else:
        pass
    return llm_model

def train_net_llm_fedlwt(net_id, net,  # target_model
                     source_optimizer, target_optimizer, wnet, lwnet, target_branch,llm_net,
                     global_net,  # source_model
                     previous_nets, train_dataloader, val_dataloader, test_dataloader, epochs,
                     lr, args_optimizer, mu, temperature, args, round, device="cpu", write_log=True):
    """
    Train the client's local model based on the HGKT and DDBL methods.
    :net_id: local model sequence number
    :net: local model
    :source_optimizer: optimizer for updating the meta-network
    :target_optimizer: optimizer for updating the local model and branch network
    :wnet, lwnet: meta-networks
    :target_branch: branch network
    :llm_net: vision foundation model
    :global_net: shared model
    :previous_nets: deprecated parameters
    :train_dataloader, val_dataloader, test_dataloader: training, validation, and test datasets
    :epochs: number of training epochs for the local model in a single communication round
    :lr: learning rate
    :args_optimizer: name of the chosen optimizer
    :mu: deprecated parameters
    :temperature: deprecated parameters
    :args: hyper-parameters
    :round: number of communication rounds
    :device: device to load the model on
    :write_log: whether to write training logs
    :returns: AUC and ACC results on training, validation, and test datasets
    """
    server_model_name = args.llm
    if write_log:
        logger.info('Training network %s' % str(net_id))
        logger.info('n_training: %d' % len(train_dataloader))
        logger.info('n_test: %d' % len(test_dataloader))
    print('Training network %s' % str(net_id))
    print('n_training: %d' % len(train_dataloader))
    print('n_test: %d' % len(test_dataloader))
    
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device,lossfunc="DiceLoss")
    if val_dataloader is not None: 
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device,lossfunc="DiceLoss")
    else:
        val_acc, val_auc = None, None
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device,lossfunc="DiceLoss")

    logger.info('>> Pre-Training Training Dice accuracy: {}--->>auc:{}'.format(train_acc, train_auc))
    logger.info('>> Pre-Training Val Dice accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    logger.info('>> Pre-Training Test Dice accuracy: {}------>>auc:{}'.format(test_acc, test_auc))
    print('>> Pre-Training Training Dice accuracy: {}-------->>auc:{}'.format(train_acc, train_auc))
    print('>> Pre-Training Val Dice accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Pre-Training Test Dice accuracy: {}------------>>auc:{}'.format(test_acc, test_auc))
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.momentum,
                              weight_decay=args.reg)
    
    if args.loss_func=="crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_func=="focalloss":
        criterion = FocalLoss(args.focalloss_alpha[net_id]).cuda(0)
    elif args.loss_func=="DiceLoss":
        criterion=DiceLoss().cuda(0)

    soft_loss = nn.KLDivLoss(reduction="batchmean")
    state = dict()
    for epoch in range(epochs):
        state['epoch'] = epoch
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_transfer_loss_collector = []
        lw_avg_batch_collector=[]
        net.train()  # local model
        global_net.eval()#shared model
        llm_net.eval()#vision foundation model
        wnet.cuda(0)#meta-network for feature channels matching
        lwnet.cuda(0)#meta-network for model layers matching
        target_branch.cuda(0)#branch network
        llm_net.cuda(0)
        for name, param in net.named_parameters(): #unfreeze all model layers
            #print(name,param.requires_grad)
            param.requires_grad = True
        for batch_idx, data in enumerate(train_dataloader):
            #Heterogeneous-model General Knowledge Transfer(HGKT)
            state['iter'] = batch_idx
            target_optimizer.zero_grad()
            loss,lw_avg = inner_objective(data, args, net, llm_net, wnet, lwnet,
                                   target_branch, state=state, logger=logger,
                                   source_model_name=server_model_name, target_model_name=net_id, device=device)
            lw_avg_batch_collector.append(lw_avg)
            loss.backward()
            target_optimizer.step(None)

            for _ in range(args.T):
                target_optimizer.zero_grad()
                target_optimizer.step(inner_objective1, data, args, net, llm_net, wnet, lwnet,
                                      target_branch, state, logger, server_model_name, net_id, True)
            target_optimizer.zero_grad()
            target_optimizer.step(outer_objective, data, args, net, state,net_id)
            target_optimizer.zero_grad()
            source_optimizer.zero_grad()
            loss = outer_objective(data, args, net, state,net_id, device=device)
            
            loss.backward()
            target_optimizer.meta_backward()
            source_optimizer.step()
            epoch_transfer_loss_collector.append(loss)
            x, target = data[0].cuda(0), data[2].cuda(0)
            out, pro1 = net(x)
            loss1 = criterion(out, target)
            
            loss1.backward()
            optimizer.step()
            epoch_loss1_collector.append(loss1.item())
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_transfer_loss = sum(epoch_transfer_loss_collector) / len(epoch_transfer_loss_collector)
        if write_log:
            logger.info('Epoch: %d  transfer_loss:%f Loss1: %f' % (epoch, epoch_transfer_loss,epoch_loss1))
        print('Epoch: %d  transfer_loss:%f Loss1: %f' % (epoch, epoch_transfer_loss,epoch_loss1))
        wnet.to('cpu')
        lwnet.to('cpu')
        target_branch.to('cpu')
        llm_net.to('cpu')
        
        # Freeze the robust critical model layer
        lw_avg_batch_collector = np.stack(lw_avg_batch_collector)
        lw_avg_batch = np.mean(lw_avg_batch_collector, axis=0)
        top_indices = np.argpartition(lw_avg_batch, -2)[-2:]
        print(lw_avg_batch,"top_layers_lw",top_indices)
        if write_log:
                logger.info("{} top_layers_lw:{}".format(lw_avg_batch,top_indices))
        for name, param in net.named_parameters():
            if name.startswith('conv1'):
                if 0 in top_indices:
                    param.requires_grad = False
            for k in top_indices:
                if name.startswith('layer%d.1'%k):
                    param.requires_grad = False
        #Data Deduction in Batch Level (DDBL)            
        if round !=0:
            global_net.cuda(0)#load shared model
            alpha=args.alpha_PI
            temp=args.temp
            for batch_idx, data in enumerate(train_dataloader):
                global_preds,_ = global_net(data[0].cuda(0))
                #model forward
                local_preds,enc= net(data[0].cuda(0))
                cross_global_preds=global_net.cross_head_forward(enc[1],enc[2],enc[3],enc[4])
                student_loss = criterion(local_preds, data[2].cuda(0))
                ditillation_loss = soft_loss(
                    F.log_softmax(cross_global_preds/temp, dim = 1),
                    F.softmax(global_preds/temp, dim = 1)
                )
                if ditillation_loss.item()>args.TV*epoch_loss1:
                    continue
                loss2 = alpha * student_loss + (1 - alpha) * ditillation_loss
                # backward
                optimizer.zero_grad()			
                loss2.backward()				
                optimizer.step()			
                epoch_loss2_collector.append(ditillation_loss)
            
            if len(epoch_loss2_collector)!=0:
                epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            else:
                epoch_loss2=0.0
            if write_log:
                logger.info('PI_loss:%f' % (epoch_loss2))
            print('PI_loss:%f' % (epoch_loss2))
    train_acc, _, train_auc = compute_accuracy(net, train_dataloader, device=device,lossfunc="DiceLoss")
    if val_dataloader is not None:
        val_acc, conf_matrix_val, _, val_auc = compute_accuracy(net, val_dataloader, get_confusion_matrix=True,
                                                                device=device,lossfunc="DiceLoss")
    test_acc, conf_matrix, _, test_auc = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                          device=device,lossfunc="DiceLoss")
    if write_log:
        logger.info('>> Training Dice accuracy: {}---->>auc:{}'.format(train_acc, train_auc))
        logger.info('>> Training Val Dice accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
        logger.info('>> Test Dice accuracy: {}----->>auc:{}'.format(test_acc, test_auc))
        logger.info(' ** Training complete **')
    print('>> Training Dice accuracy: %f------->>auc:%f' % (train_acc, train_auc))
    print('>> Training Val Dice accuracy: {}----->>auc:{}'.format(val_acc, val_auc))
    print('>> Test Dice accuracy: %f----------->>auc:%f' % (test_acc, test_auc))
    net.to('cpu')
    global_net.to('cpu')
    print(' ** Training complete **')
    return train_acc, val_acc, test_acc, train_auc, val_auc, test_auc

def local_train_net(nets, args,
                    source_optimizers_dict=None, target_optimizers_dict=None,
                    wnet_dict=None, lwnet_dict=None, target_branch_dict=None,
                    train_dl=None, val_dl=None, test_dl=None,llm_model=None,
                    global_model=None, prev_model_pool=None, server_c=None, clients_c=None, round=None, device="cpu",
                    write_log=True):
    """
    Train models on specified architectures for each client:
    :nets: All local models
    :args: Hyper-parameters
    :source_optimizer_dict: Optimizers for updating the meta-network
    :target_optimizer_dict: Optimizers for updating local models and branch networks
    :wnet_dict, lwnet_dict: All meta-networks
    :target_branch: All branch networks
    :train_dl, val_dl, test_dl: Training, validation, and test datasets for all centers
    :llm_model: Vision foundation model
    :global_model: Shared model
    :prev_model_pool, server_c, clients_c: Deprecated parameters
    :round: Communication rounds
    :device=Select device for loading models
    :write_log: Choose whether to write training logs
    :returns: Local models and AUC results for each client
    """
    if args.alg == "fedlwt":#only for fedlwt
        local_models = args.sites.copy()
        server_model_name = [args.sites[args.global_center_idx]][0]
        local_models.pop(args.global_center_idx)
    else:#for llm_fedlwt
        local_models = args.sites.copy()
        server_model_name = args.llm

    avg_acc = 0.0
    acc_list = []
    auc = {}

    for idx, (net_id, net) in enumerate(nets.items()):
        if write_log:
            logger.info("Training network %s. batch_id: %s" % (str(net_id), str(net_id)))
        print("Training network %s. batch_id: %s" % (str(net_id), str(net_id)))
        train_dl_local = train_dl[idx]
        if val_dl is not None:
            val_dl_local = val_dl[idx]
        else:
            val_dl_local = None
        test_dl_local = test_dl[idx]
        n_epoch = args.epochs

        if args.alg == 'llm_fedlwt':
            #Select optimizers, meta-networks, local models, and branch networks for respective clients.
            source_optimizer = source_optimizers_dict["%sto%s" % (server_model_name, net_id)]
            target_optimizer = target_optimizers_dict["%sto%s" % (server_model_name, net_id)]
            net=net.cuda(0)
            wnet = wnet_dict["%sto%s" % (server_model_name, net_id)]
            lwnet = lwnet_dict["%sto%s" % (server_model_name, net_id)]
            target_branch = target_branch_dict["%sto%s" % (server_model_name, net_id)]
            ##############
            prev_models = []#deprecated parameters
            # for i in range(len(prev_model_pool)):
            #     prev_models.append(prev_model_pool[i][net_id])
            train_acc, val_acc, test_acc, train_auc, val_auc, test_auc = train_net_llm_fedlwt(net_id, net,
                                                                                          source_optimizer,
                                                                                          target_optimizer,
                                                                                          wnet, lwnet, target_branch,llm_model,
                                                                                          global_model, prev_models,
                                                                                          train_dl_local, val_dl_local,
                                                                                          test_dl_local,
                                                                                          n_epoch, args.lr,
                                                                                          args.optimizer, args.mu,
                                                                                          args.temperature, args, round,
                                                                                          device=device,
                                                                                          write_log=write_log)
            auc[net_id] = [train_auc, test_auc]
            
        if write_log:
            logger.info("net %s final test acc %f" % (net_id, test_acc))
        print("net %s final test acc %f" % (net_id, test_acc))
        avg_acc += test_acc
        acc_list.append(test_acc)
    avg_acc /= len(args.sites)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    return nets, auc


if __name__ == '__main__':
    #hyper-parameters setting
    #logger setting and device setting
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    #load data#
    if args.alg == 'llm_fedlwt':
        sites, _, _, train_loaders, val_loaders, test_loaders, net_dataidx_map=recorded_llm_multicenters_dataloader(args)
    ######
    args.sites = sites
    if len(val_loaders) == 0:
        val_dl_global = None
        val_loaders = None
    #n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = sites
    party_list_rounds = []
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

    # local model,vision foundation model and shared model initialization
    logger.info("Initializing nets")
    print("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args, device='cpu')
    llm_model=llm_init(args.llm)
    global_models, global_model_meta_data, global_layer_type = init_nets(args, device='cpu', server=True)
    global_model = global_models["server"]


    if args.alg == 'llm_fedlwt':
        #meta networks initialization
        pairs = init_pairs(args)
        wnet_dict, lwnet_dict, weight_params_dict, target_params_dict, target_branch_dict = init_meta_model(args, pairs,
                                                                                                            global_model,
                                                                                                            nets)  # nets or global_model

        source_optimizers_dict, target_optimizers_dict = optimizer_init(args, weight_params_dict, nets,
                                                                        target_params_dict, target_branch_dict)

        old_nets_pool = []#deprecated
        #train
        for round in range(args.comm_round):
            time1 = time.time()
            logger.info("in comm round:" + str(round))
            print("in comm round:" + str(round))

            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            
            _, local_auc = local_train_net(nets_this_round, args,
                                           source_optimizers_dict=source_optimizers_dict,
                                           target_optimizers_dict=target_optimizers_dict,
                                           wnet_dict=wnet_dict, lwnet_dict=lwnet_dict,
                                           target_branch_dict=target_branch_dict,
                                           train_dl=train_loaders, val_dl=val_loaders, test_dl=test_loaders,llm_model=llm_model,
                                           global_model=global_model, prev_model_pool=old_nets_pool, round=round,
                                           device=device, write_log=True)
            #shared model aggregation
            client_num = len(party_list_this_round)
            fed_avg_freqs = [1. / client_num for i in range(client_num)]
            print(fed_avg_freqs)
            for key in global_model.state_dict().keys():
                temp = torch.zeros_like(global_model.state_dict()[key],dtype=torch.float32)
                for net_id, net in enumerate(nets_this_round.values()):
                    temp += fed_avg_freqs[net_id] * net.state_dict()[key]
                global_model.state_dict()[key].data.copy_(temp)
            #save model
            if round % args.save_model_interval == 0:
                if args.save_model:
                    mkdirs(args.modeldir + '%s/' % args.alg)
                    torch.save(global_model.state_dict(),
                               args.modeldir + '%s/global_model_%d' % (args.alg, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(nets):
                        print(net_id,"model_saved")
                        torch.save(nets[net_id].state_dict(), args.modeldir + '%s/localmodel_%s_%d' % (
                        args.alg, net_id, round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(wnet_dict):
                        torch.save(wnet_dict[net_id].state_dict(), args.modeldir + '%s/wnet_model_%s_%d' % (args.alg,
                                                                                                            net_id,
                                                                                                            round) + args.log_file_name + '.pth')
                    for k, net_id in enumerate(lwnet_dict):
                        torch.save(lwnet_dict[net_id].state_dict(), args.modeldir + '%s/lwnet_model_%s_%d' % (args.alg,
                                                                                                              net_id,
                                                                                                              round) + args.log_file_name + '.pth')
            time2 = time.time()
            print("round:%d,consume:%s minutes" % (round, (time2 - time1) / 60.0))

