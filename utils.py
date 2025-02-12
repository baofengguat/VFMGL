import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix,roc_auc_score
#from HarmoFL_utils.layers import AmpNorm
from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom
from torch.utils import data
from lw2w import DiceLoss,JointLoss
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# amp_norm=AmpNorm((3,224,224))
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_n_center_lung_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    center_set={}

    for center_name in os.listdir(datadir):
        sub_set = {}
        if center_name == ".DS_Store" or center_name == "A":
            continue
        center_path=os.path.join(datadir,center_name)

        center_train_ds = ImageFolder_custom(center_path + '/train_data/', transform=transform)
        center_test_ds = ImageFolder_custom(center_path + '/test_data/', transform=transform)

        X_train, y_train = np.array([s[0] for s in center_train_ds.samples]), np.array(
            [int(s[1]) for s in center_train_ds.samples])
        X_test, y_test = np.array([s[0] for s in center_test_ds.samples]), np.array(
            [int(s[1]) for s in center_test_ds.samples])

        sub_set["X_train"],sub_set["y_train"]=X_train,y_train
        sub_set["X_test" ], sub_set["y_test" ] = X_test, y_test
        center_set["%s" % center_name] =sub_set
        print("%s_center_loaded_finished"%center_name)
    return center_set
def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'./train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'./val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)
def load_server_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'./train_data/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'./test_data/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)
def record_net_multicenter_data_stats(y_train_set, net_dataidx_map, logdir):
    net_cls_counts = {}
    # print(net_dataidx_map.items())
    for net_i, dataidx in net_dataidx_map.items():
        # print(dataidx)
        y_train=np.array(y_train_set[net_i])
        # print(y_train)

        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts
def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}
    # print(net_dataidx_map.items())
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset=='lung_nodules' or dataset=='lung_nodules_amp':
        sets=load_n_center_lung_data(datadir)
    elif dataset=="camelyon17":
        return
    if dataset in ['cifar10','cifar100','tinyimagenet']:
        n_train = y_train.shape[0]
        if partition == "homo" or partition == "iid":
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, n_parties)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        elif partition == "noniid-labeldir" or partition == "noniid":
            min_size = 0
            min_require_size = 10
            K = 10
            if dataset == 'cifar100':
                K = 100
            elif dataset == 'tinyimagenet':
                K = 200
                # min_require_size = 100
            N = y_train.shape[0]
            net_dataidx_map = {}
            while min_size < min_require_size:
                idx_batch = [[] for _ in range(n_parties)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                    proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    # if K == 2 and n_parties <= 10:
                    #     if np.min(proportions) < 200:
                    #         min_size = 0
                    #         break
            for j in range(n_parties):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
        return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
    else:
        X_train_set = []
        y_train_set = {}
        X_test_set = []
        y_test_set = []
        net_dataidx_map_set={}
        traindata_cls_counts_set=[]
        # print(sets.keys())
        for key,set in sets.items():
            X_train, y_train, X_test, y_test=set["X_train"],set["y_train"],set["X_test"],set["y_test"]
            n_train = y_train.shape[0]
            if partition == "homo" or partition == "iid":
                idxs = np.random.permutation(n_train)
            elif partition == "noniid-labeldir" or partition == "noniid":
                min_size = 0
                min_require_size = 10
                K = 100
                    # min_require_size = 100
                N = y_train.shape[0]
                net_dataidx_map = {}
                while min_size < min_require_size:
                    idx_batch = [[] for _ in range(n_parties)]
                    for k in range(K):
                        idx_k = np.where(y_train == k)[0]
                        np.random.shuffle(idx_k)
                        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                        proportions = np.array(
                            [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                        proportions = proportions / proportions.sum()
                        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                        idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                                     zip(idx_batch, np.split(idx_k, proportions))]
                        min_size = min([len(idx_j) for idx_j in idx_batch])
                        # if K == 2 and n_parties <= 10:
                        #     if np.min(proportions) < 200:
                        #         min_size = 0
                        #         break
                for j in range(n_parties):
                    np.random.shuffle(idx_batch[j])
                    net_dataidx_map[j] = idx_batch[j]
            # traindata_cls_counts = record_net_data_stats(y_train, n_train, logdir)
            X_train_set.append(X_train)
            # y_train_set.extend(y_train.tolist())
            y_train_set[key]=y_train.tolist()
            X_test_set.append(X_test)
            y_test_set.append(y_test)
            net_dataidx_map_set[key]=idxs.tolist() #存疑
            # print(key,len(y_train.tolist()))
        # print(len((y_train_set)))
        # traindata_cls_counts=record_net_data_stats(((y_train_set)), net_dataidx_map_set, logdir)
        traindata_cls_counts = record_net_multicenter_data_stats((y_train_set), net_dataidx_map_set, logdir)
        return(X_train_set,y_train_set,X_test_set,y_test_set,net_dataidx_map_set,traindata_cls_counts)


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False,lossfunc=None):
    was_training = False
    label_list=[]
    pred_list=[]

    if model.training: 
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0., 0
    avg_auc=0

    if lossfunc=="DiceLoss":
        if "cuda" in device.type:
            criterion=DiceLoss().cuda(0)
        else:
            criterion = DiceLoss()
    elif lossfunc=="JointLoss":
        if "cuda" in device.type:
            criterion=JointLoss().cuda(0)
        else:
            criterion = JointLoss()
    else:
        if "cuda" in device.type:
            # print("cuda")
            criterion = nn.CrossEntropyLoss().cuda(0)
        else:
            criterion = nn.CrossEntropyLoss()
    loss_collector = []
    # criterion=nn.CrossEntropyLoss()
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda().long()
                    _, _, out = model(x)
                    if len(target)==1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, data1 in enumerate(dataloader):
                #print("x:",x)de
                if device != 'cpu':
                    x, target = data1[0].cuda(0), data1[-1].to(dtype=torch.int64).cuda(0).long()
                out,features = model(x)
                # print(out,target)

                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                # print(out.data.cpu().numpy())
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                if lossfunc in ["DiceLoss", "JointLoss"]:
                    correct += DiceLoss().dice_coef(out, target).item()
                else:
                    correct += (pred_label == target.data).sum().item()
                #auc
                label_list.extend(target.cpu().numpy().tolist())

                pred_list.extend(out.data.cpu().numpy()[:,1])
                # batch_auc = roc_auc_score(target.cpu().numpy(), pred_label.cpu().numpy())

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)
            if lossfunc is not None:
                avg_auc=0.0
            else:
                try:
                    avg_auc=roc_auc_score(label_list,pred_list)
                except:
                    avg_auc=0.0
            # print("auc：",avg_auc)
    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
        
    if was_training:
        model.train()
    if lossfunc in ["DiceLoss", "JointLoss"]:
        if get_confusion_matrix:
            return correct / len(dataloader), conf_matrix, avg_loss,avg_auc

        return correct / len(dataloader), avg_loss,avg_auc
    else:
        if get_confusion_matrix:
            return correct / float(total), conf_matrix, avg_loss, avg_auc

        return correct / float(total), avg_loss, avg_auc


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda(0)
    return model


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0,center_name="江门医院"):

    if dataset=="lung_nodules" or dataset=="lung_nodules_amp":
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # AmpNorm((3,224,224))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # AmpNorm((3,224,224))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir + '/%s/train_data/'%center_name, dataidxs=dataidxs, transform=transform_train)#server训练数据默认为江门医院
        test_ds = dl_obj(datadir + '/%s/test_data/'%center_name, transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True,num_workers=8,pin_memory=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,num_workers=8,pin_memory=True)
    return train_dl, test_dl, train_ds, test_ds
