import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os
from packaging import version
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from l2t_ww.check_model import check_model
from N_data_dataloaders import *
from lw2w import DiceLoss,JointLoss
from PIL import Image
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from l2t_ww.models import resnet_ilsvrc
from utils import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
MODEL = 'resnet18'
model_name="resnet18"
dataset="camelyon17"
fed_structure="llm_fedlwt"
DATA_DIRECTORY = r"./data/%s"%dataset
NUM_CLASSES = 2
RESTORE_FROM = r'./model_weights/camelyon17'
PRETRAINED_MODEL = None
global_index=None #0 or None
batchsize=32#use the same batch size as during model training
if dataset=="camelyon17":
    center_names=['1', '2', '3', '4', '5']

def get_arguments():
    parser = argparse.ArgumentParser(description="evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--model_name", type=str, default=model_name,
                        help="available options : resnet18")
    parser.add_argument("--datadir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data-list", type=str, default=None,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset_train used for training')
    parser.add_argument('--imbalance', type=bool, default=True, help='do not truncate train data to same length')
    parser.add_argument('--batch-size', type=int, default=batchsize, help='input batch size for training (default: 64)')
    return parser.parse_args()

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    device = torch.device("cuda:0")
    
    sites, _, _, train_loaders, val_loaders, test_loaders, net_dataidx_map = \
        recorded_multicenters_data_API_dataloader(args)
    set_list=["train","val","test"]#,
    ResNet = resnet_ilsvrc.__dict__[args.model_name]  # (pretrained=True)
    model = ResNet(num_classes=2, mhsa=False, dropout_p=0,
                resolution=(224, 224), open_perturbe=False)
    
    for set_name in set_list:
        #set_name="val"
        print(set_name)
        if set_name=="train":
            loaders=train_loaders
        elif set_name=="val":
            loaders=val_loaders
        elif set_name=="test":
            loaders=test_loaders
        for index1,loader in enumerate(loaders):
            center_name=center_names[index1]
            model_name = "localmodel_" + center_name+".pth"
            print("===%s===predict===%s==="%(center_name,center_name))
            trained_model_path = os.path.join(args.restore_from, model_name)
            saved_state_dict = torch.load(trained_model_path)
            model.load_state_dict(saved_state_dict)
            model.eval()
            model.cuda()
            acc, conf_matrix,_, auc = compute_accuracy(model, loader, device=device,get_confusion_matrix=True)
            
            TN, FP, FN, TP = conf_matrix.ravel()
            Specificity = TN / (TN + FP)
            Sensitivity = TP / (TP + FN)
            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            PPV = TP / (TP + FP)
            NPV = TN / (TN + FN)
            print('AUC=%f,Sensitivity=%f(%d/%d),Specificity=%f(%d/%d),acc=%f(%d/%d),PPV=%3f(%d/%d),NPV=%3f(%d/%d)'% (auc,Sensitivity,TP,TP+FN,Specificity,TN,FP+TN,Accuracy,TP+TN,TP+FP+FN+TN,PPV,TP,TP+FP,NPV,TN,TN+FN))
            print("positive_num:%d,negative_num:%d"%(TP+FN,TN+FP))
         
            
if __name__ == '__main__':
    main()
