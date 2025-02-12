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

MODEL = 'UNet'
model_name="unet"
dataset="nuclei"
fed_structure="llm_fedlwt"
DATA_DIRECTORY = r"./data/%s"%dataset
IGNORE_LABEL = 255
NUM_CLASSES = 2
RESTORE_FROM = r'./model_weights/nuclei'
PRETRAINED_MODEL = None
#When calculating Dice accuracy, use the same batch size as during model training.
#For calculating metrics such as ASSD, due to third-party library limitations, the batchsize needs to be adjusted to 1.
batchsize=2 
if dataset=="nuclei":
    center_names=['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']


def get_arguments():
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--model_name", type=str, default=model_name,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--datadir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=None,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
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
# batch_size = 1
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

from medpy.metric.binary import hd95,recall,sensitivity,specificity,\
    true_positive_rate,true_negative_rate,precision,positive_predictive_value
import surface_distance as surfdist
def compute_predictor(predict,groundtruth):
    HD95=hd95(predict,groundtruth)
    RECALL=recall(predict,groundtruth)
    Sensi=sensitivity(predict,groundtruth)
    Speci=specificity(predict,groundtruth)
    TPR=true_positive_rate(predict,groundtruth)
    TNR=true_negative_rate(predict,groundtruth)
    preci=precision(predict,groundtruth)
    PPV=positive_predictive_value(predict,groundtruth)
    #groundtruth[np.where(groundtruth == 1)] = True
    #groundtruth[np.where(groundtruth == 0)] = False
    IOU=iou_score(predict,groundtruth)
    surface_distances = surfdist.compute_surface_distances(
        groundtruth.astype(np.bool_), predict.astype(np.bool_), spacing_mm=(1.0, 1.0))
    asd_dict = surfdist.compute_average_surface_distance(surface_distances)
    assd=np.mean(asd_dict)
    #print(asd_dict,assd)
    return HD95,assd,RECALL,Sensi,Speci,TPR,TNR,preci,PPV,IOU

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    device="cuda"
    
    sites, _, _, train_loaders, val_loaders, test_loaders, net_dataidx_map = \
        recorded_multicenters_data_API_dataloader(args)
    set_list=["train","val","test"]
    model =check_model(args).to(device)
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
            # print(model_name)
            trained_model_path = os.path.join(args.restore_from, model_name)
            saved_state_dict = torch.load(trained_model_path)
            model.load_state_dict(saved_state_dict)
            model.eval()
            if device=="cuda":
                model.cuda()
            else:
                model.cpu()
        
            img_num=len(loader)
            HD951, RECALL1, Sensi1, Speci1, TPR1, TNR1, preci1, PPV1,IOU1=0,0,0,0,0,0,0,0,0
            assd1=0
            correct=0
            for index, batch in enumerate(loader):
                image, label,name= batch
                if device == "cuda":
                    output,_ = model(image.cuda())
                else:
                    output,_ = model(image).cpu()
                #x.cuda(0), target.to(dtype=torch.int64).cuda(0).long()
                correct += DiceLoss().dice_coef(output, label.to(dtype=torch.int64).cuda().long()).item()
                gt=label[0].numpy()
                output = output.cpu().data[0].numpy().transpose(1,2,0)
                output1=np.asarray(np.argmax(output, axis=2), dtype=np.int16)
                if batchsize==1:
                    HD95, assd, RECALL, Sensi, Speci, TPR, TNR, preci, PPV,IOU=compute_predictor(output1,gt[0])
                    HD951+=HD95
                    RECALL1+=RECALL
                    Sensi1+=Sensi
                    Speci1+=Speci
                    TPR1+=TPR
                    TNR1+=TNR
                    preci1+=preci
                    PPV1+=PPV
                    assd1+=assd
                    IOU1+=IOU
            if batchsize==1:
                print("HD95:%f ASSD:%f RECALL:%f Sensitivity:%f Specitity:%f"%(HD951/img_num, assd1/img_num, RECALL1/img_num, Sensi1/img_num, Speci1/img_num))
                print("TPR:%f TNR:%f precision:%f PPV:%f" % (TPR1/img_num, TNR1/img_num, preci1/img_num, PPV1/img_num))
                print("IOU:%f"%(IOU1/img_num))
            if batchsize==2:
                print("Dice:%f"%(correct/img_num))
            


if __name__ == '__main__':
    main()
