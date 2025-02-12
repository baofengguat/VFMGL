'''
use for multi-center data loading
the data includes:
camelyon17/prostate/Nuclei/EC
'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import SimpleITK as sitk
import random
import cv2
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.datasets import ImageFolder, DatasetFolder, CIFAR10, CIFAR100
import math
import logging
import shutil
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

####API
class Camelyon17(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'test']
        assert int(site) in [1, 2, 3, 4, 5]  # five hospital

        base_path = base_path if base_path is not None else 'E:\lusl\external_data\camelyon17'
        self.base_path = base_path

        data_dict = np.load(os.path.join(base_path,'data.pkl'), allow_pickle=True)
        self.paths, self.labels = data_dict[f'hospital{site}'][f'{split}']

        self.transform = transform
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))
        if self.transform is not None:
            image = self.transform(image)

        return image, label

class Prostate(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'ISBI': 3, 'ISBI_1.5': 3, 'UCL': 3}
        assert site in list(channels.keys())
        self.split = split

        base_path = base_path if base_path is not None else '../data/prostate'

        images, labels = [], []
        sitedir = os.path.join(base_path, site)
        img_names=[]
        img_s=[]
        ossitedir = np.load(os.path.join(base_path,"{}-dir.npy".format(site))).tolist()
        window_width=300
        window_center=50
        for sample in ossitedir:
            sampledir = os.path.join(sitedir, sample)
            if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
                label_v = sitk.ReadImage(sampledir)
                image_v = sitk.ReadImage(imgdir)
                label_v = sitk.GetArrayFromImage(label_v)
                label_v[label_v > 1] = 1
                image_v = sitk.GetArrayFromImage(image_v)
                # image_s= np.clip((image_v.copy() - (window_center - window_width / 2)) / window_width * 255, 0, 255).astype(np.uint8)
                image_v = convert_from_nii_to_png(image_v)
                
                for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if (np.all(label == 0)):
                        continue
                    image = np.array(image_v[i - 1:i + 2, :, :])
                    image = np.transpose(image, (1, 2, 0))
                    img_names.append("%s_%d"%(sample[:6],i))
                    labels.append(label)
                    images.append(image)
                    # img_s.append(image_s[i])
        labels = np.array(labels).astype(int)
        images = np.array(images)
        # img_s=np.array(img_s)
        img_names=np.array(img_names)
        index = np.load(os.path.join(base_path,"{}-index.npy".format(site))).tolist()

        labels = labels[index]
        images = images[index]
        # img_s=img_s[index]
        img_names=img_names[index]
        trainlen = 0.8 * len(labels) * 0.8
        vallen = 0.8 * len(labels) - trainlen
        testlen = 0.2 * len(labels)

        if (split == 'train'):
            self.images, self.labels = images[:int(trainlen)], labels[:int(trainlen)]
            self.img_names=img_names[:int(trainlen)]
            # self.img_s=img_s[:int(trainlen)]
        elif (split == 'val'):
            self.images, self.labels = images[int(trainlen):int(trainlen + vallen)], labels[int(trainlen):int(
                trainlen + vallen)]
            self.img_names=img_names[int(trainlen):int(trainlen + vallen)]
            # self.img_s=img_s[int(trainlen):int(trainlen + vallen)]
        else:
            self.images, self.labels = images[int(trainlen + vallen):], labels[int(trainlen + vallen):]
            self.img_names=img_names[int(trainlen + vallen):]
            # self.img_s=img_s[int(trainlen + vallen):]
        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.int64).squeeze()
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image, (2, 0, 1))
            image = torch.Tensor(image)

            label = self.transform(label)

        return image, label,self.img_names[idx]##,self.img_s[idx]

class Nuclei(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'val', 'test']
        assert site in ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']

        self.base_path = base_path if base_path is not None else r'E:\\lusl\\external_data\\NucSeg'
        self.base_path = os.path.join(self.base_path, site)

        images = []
        labels = []
        img_names=[]
        if split == 'train' or split == 'val':
            self.base_path = os.path.join(self.base_path, split)
            img_path = os.path.join(self.base_path, "images")
            lbl_path = os.path.join(self.base_path, "labels")
            for i in os.listdir(img_path):
                if i.startswith("._"):
                    continue
                img_dir = os.path.join(img_path, i)
                ibl_dir = os.path.join(lbl_path, i.split('.')[0] + ".png")
                images.append(img_dir)
                labels.append(ibl_dir)
                img_names.append(i)
            self.images, self.labels,self.img_names = images, labels,img_names

        elif split == 'test':
            self.base_path = os.path.join(self.base_path, "test")
            img_path = os.path.join(self.base_path, "images")
            lbl_path = os.path.join(self.base_path, "labels")
            for i in os.listdir(img_path):
                if i.startswith("._"):
                    continue
                img_dir = os.path.join(img_path, i)
                ibl_dir = os.path.join(lbl_path, i.split('.')[0] + ".png")
                images.append(img_dir)
                labels.append(ibl_dir)
                img_names.append(i)
            self.images, self.labels,self.img_names = images, labels,img_names

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = cv2.imread(self.labels[idx], 0)
        image = Image.open(self.images[idx].replace("\\", "/")).convert('RGB')

        label[label == 255] = 1

        label = Image.fromarray(label)

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

            TTensor = transforms.ToTensor()
            image = TTensor(image)

            label = np.array(label)
            label = torch.Tensor(label)

            label = torch.unsqueeze(label, dim=0)

        return image, label,self.img_names[idx]#

#####functional function
def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

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

class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()

class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return img, mask

def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0,center_name=None):
    if dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])



        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir+'./train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'./val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    # elif dataset=="lung_nodules" or dataset=="lung_nodules_amp":
    #     dl_obj = ImageFolder_custom
    #     transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         # AmpNorm((3,224,224))
    #     ])
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         # AmpNorm((3,224,224))
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #
    #     train_ds = dl_obj(datadir + '/%s/train_data/'%center_name, dataidxs=dataidxs, transform=transform_train)#server训练数据默认为江门医院
    #     test_ds = dl_obj(datadir + '/%s/test_data/'%center_name, transform=transform_test)
    #
    #     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True,num_workers=8,pin_memory=True)
    #     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,num_workers=8,pin_memory=True)
    #
    # elif dataset == "gastric":
    #     dl_obj = ImageFolder_custom
    #     transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize([224, 224])
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         # AmpNorm((3,224,224))
    #     ])
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize([224, 224])
    #         # AmpNorm((3,224,224))
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #
    #     train_ds = dl_obj(datadir + '/%s/train_data/' % center_name, dataidxs=dataidxs,
    #                       transform=transform_train)  # server训练数据默认为江门医院
    #     test_ds = dl_obj(datadir + '/%s/test_data/' % center_name, transform=transform_test)
    #
    #     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=8,
    #                                pin_memory=True)
    #     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)
    # elif dataset == "lidc":
    #     dl_obj = ImageFolder_custom
    #     transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize([224, 224])
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         # AmpNorm((3,224,224))
    #     ])
    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Resize([224, 224])
    #         # AmpNorm((3,224,224))
    #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    #
    #     train_ds = dl_obj(datadir + '/%s/train_data/' % center_name, dataidxs=dataidxs,
    #                       transform=transform_train)  # server训练数据默认为江门医院
    #     test_ds = dl_obj(datadir + '/%s/test_data/' % center_name, transform=transform_test)
    #
    #     train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=8,
    #                                pin_memory=True)
    #     test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=8, pin_memory=True)
    return train_dl, test_dl, train_ds, test_ds


class med_DataSet(data.Dataset):
    def __init__(self, root=None,
                 data_set="",center="",transform=None,patch_h=112,patch_w=112):
        self.root = root
        self.files = []
        self.patient_img_num=[]
        self.transform=transform

        img_num=0
        leison1_list=["肺腺癌","1","复发","进展"]#针对肺结节/胃复发/LIDC整理数据
        for leison_class in os.listdir(os.path.join(self.root,center,data_set)):
            for patient in os.listdir(os.path.join(self.root,center,data_set,leison_class)):
            # for split in ["train", "trainval", "val"]:
                for img_name in os.listdir(os.path.join(self.root,center,data_set,leison_class,patient)):
                    if ".baiduyun" in img_name:
                        continue
                    img_path=os.path.join(
                                self.root, center, data_set, leison_class, patient,img_name)
                    self.files.append({
                        "img_path": img_path,
                        "label": 1 if leison_class in leison1_list else 0,
                        "center": center,
                        "leison_class": leison_class,
                        "patient_name": patient})
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        # patient_name = datafiles["patient_name"]
        # center_name = datafiles["center"]
        # leisonclass = datafiles["leison_class"]
        label = datafiles["label"]
        image = Image.open(datafiles["img_path"]).convert('RGB')

        image=image.resize((224,224))
        if self.transform is not None:
            image = self.transform(image)
        return image,label


class llm_med_DataSet(data.Dataset):
    def __init__(self, root=None,
                 data_set="",center="",transform=None,patch_h=112,patch_w=112):
        self.root = root
        self.files = []
        self.patient_img_num=[]
        self.transform=transform
        self.transform1 = transforms.Compose([
            # transforms.GaussianBlur(9, sigma=(0.1, 2.0)),
            transforms.Resize((patch_h * 14, patch_w * 14)),
            transforms.CenterCrop((patch_h * 14, patch_w * 14)),
            transforms.ToTensor(),
            # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        img_num=0
        leison1_list=["肺腺癌","1","复发","进展"]#针对肺结节/胃复发/LIDC整理数据
        for leison_class in os.listdir(os.path.join(self.root,center,data_set)):
            for patient in os.listdir(os.path.join(self.root,center,data_set,leison_class)):
            # for split in ["train", "trainval", "val"]:
                for img_name in os.listdir(os.path.join(self.root,center,data_set,leison_class,patient)):
                    if ".baiduyun" in img_name:
                        continue
                    img_path=os.path.join(
                                self.root, center, data_set, leison_class, patient,img_name)
                    self.files.append({
                        "img_path": img_path,
                        "label": 1 if leison_class in leison1_list else 0,
                        "center": center,
                        "leison_class": leison_class,
                        "patient_name": patient})
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        label = datafiles["label"]
        image = Image.open(datafiles["img_path"]).convert('RGB')
        image=image.resize((224,224))
        llm_image=self.transform1(image)
        if self.transform is not None:
            image = self.transform(image)

        return image,llm_image,label

class llm_Camelyon17(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None,patch_h=112,patch_w=112):
        assert split in ['train', 'test']
        assert int(site) in [1, 2, 3, 4, 5]  # five hospital

        base_path = base_path if base_path is not None else 'E:\lusl\external_data\camelyon17'
        self.base_path = base_path

        data_dict = np.load(os.path.join(base_path,'data.pkl'), allow_pickle=True)
        self.paths, self.labels = data_dict[f'hospital{site}'][f'{split}']

        self.transform = transform
        self.labels = self.labels.astype(np.int64).squeeze()
        self.transform1 = transforms.Compose([
            # transforms.GaussianBlur(9, sigma=(0.1, 2.0)),
            transforms.Resize((patch_h * 14, patch_w * 14)),
            transforms.CenterCrop((patch_h * 14, patch_w * 14)),
            transforms.ToTensor(),
            # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))
        llm_image=self.transform1(image)
        if self.transform is not None:
            image = self.transform(image)

        return image,llm_image,label

class llm_Nuclei(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None,patch_h=112,patch_w=112):
        assert split in ['train', 'val', 'test']
        assert site in ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']

        self.base_path = base_path if base_path is not None else r'E:\\lusl\\external_data\\NucSeg'
        self.base_path = os.path.join(self.base_path, site)
        self.transform1 = transforms.Compose([
            transforms.Resize((patch_h * 14, patch_w * 14)),
            transforms.CenterCrop((patch_h * 14, patch_w * 14)),
            transforms.ToTensor()])
        images = []
        labels = []

        if split == 'train' or split == 'val':
            self.base_path = os.path.join(self.base_path, split)
            img_path = os.path.join(self.base_path, "images")
            lbl_path = os.path.join(self.base_path, "labels")
            for i in os.listdir(img_path):
                if i.startswith("._"):
                    continue
                img_dir = os.path.join(img_path, i)
                ibl_dir = os.path.join(lbl_path, i.split('.')[0] + ".png")
                images.append(img_dir)
                labels.append(ibl_dir)

            self.images, self.labels = images, labels

        elif split == 'test':
            self.base_path = os.path.join(self.base_path, "test")
            img_path = os.path.join(self.base_path, "images")
            lbl_path = os.path.join(self.base_path, "labels")
            for i in os.listdir(img_path):
                if i.startswith("._"):
                    continue
                img_dir = os.path.join(img_path, i)
                ibl_dir = os.path.join(lbl_path, i.split('.')[0] + ".png")
                images.append(img_dir)
                labels.append(ibl_dir)

            self.images, self.labels = images, labels

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = cv2.imread(self.labels[idx], 0)
        image = Image.open(self.images[idx].replace("\\", "/")).convert('RGB')

        label[label == 255] = 1

        label = Image.fromarray(label)

        if self.transform is not None:
            llm_image=self.transform1(image)
            image = self.transform(image)
            label = self.transform(label)

            TTensor = transforms.ToTensor()
            
            image = TTensor(image)
            
            label = np.array(label)
            label = torch.Tensor(label)

            label = torch.unsqueeze(label, dim=0)

        return image,llm_image,label

class llm_Prostate(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None,patch_h=64,patch_w=64):
        channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'ISBI': 3, 'ISBI_1.5': 3, 'UCL': 3}
        assert site in list(channels.keys())
        self.split = split
        self.transform1 = transforms.Compose([
            transforms.Resize((patch_h * 14, patch_w * 14)),
            transforms.CenterCrop((patch_h * 14, patch_w * 14)),
            transforms.ToTensor()])
        base_path = base_path if base_path is not None else '../data/prostate'
        images, labels = [], []
        sitedir = os.path.join(base_path, site)

        ossitedir = np.load(os.path.join(base_path,"{}-dir.npy".format(site))).tolist()

        for sample in ossitedir:
            sampledir = os.path.join(sitedir, sample)
            if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
                label_v = sitk.ReadImage(sampledir)
                image_v = sitk.ReadImage(imgdir)
                label_v = sitk.GetArrayFromImage(label_v)
                label_v[label_v > 1] = 1
                image_v = sitk.GetArrayFromImage(image_v)
                image_v = convert_from_nii_to_png(image_v)

                for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if (np.all(label == 0)):
                        continue
                    image = np.array(image_v[i - 1:i + 2, :, :])
                    image = np.transpose(image, (1, 2, 0))

                    labels.append(label)
                    images.append(image)
        labels = np.array(labels).astype(int)
        images = np.array(images)

        index = np.load(os.path.join(base_path,"{}-index.npy".format(site))).tolist()

        labels = labels[index]
        images = images[index]

        trainlen = 0.8 * len(labels) * 0.8
        vallen = 0.8 * len(labels) - trainlen
        testlen = 0.2 * len(labels)

        if (split == 'train'):
            self.images, self.labels = images[:int(trainlen)], labels[:int(trainlen)]

        elif (split == 'val'):
            self.images, self.labels = images[int(trainlen):int(trainlen + vallen)], labels[int(trainlen):int(
                trainlen + vallen)]
        else:
            self.images, self.labels = images[int(trainlen + vallen):], labels[int(trainlen + vallen):]

        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)
            img_pil = Image.fromarray(image)
            image = np.transpose(image, (2, 0, 1))
            
            llm_image=self.transform1(img_pil)
            image = torch.Tensor(image)

            label = self.transform(label)

        return image,llm_image, label

def recorded_llm_multicenters_dataloader(args):
    publish_data1 = ["camelyon17", "prostate", "nuclei"]
    private_data = ["Endometrial_Cancer"]
    train_loaders, test_loaders = [], []
    val_loaders = []
    trainsets, testsets = [], []
    valsets = []
    sites = []
    if args.dataset in private_data:
        if args.dataset=="Endometrial_Cancer":
            net_dataidx_map = {}
            sites = ["A", "B", "C","D"]#
            for site in sites:
                train_set = llm_med_DataSet(root=args.datadir, center=site, data_set="traindata",patch_h=64,patch_w=64,transform=transforms.ToTensor())
                test_set = llm_med_DataSet(root=args.datadir, center=site, data_set="testdata",patch_h=64,patch_w=64,transform=transforms.ToTensor())
                train_dl_local = data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,num_workers=args.num_workers,pin_memory=True)
                test_dl_local = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,pin_memory=True)
                train_loaders.append(train_dl_local)
                test_loaders.append(test_dl_local)
                net_dataidx_map[site] = train_set
    if args.dataset in publish_data1:
        if args.dataset == 'camelyon17':
           
            sites = ['1', '2', '3', '4', '5']
            net_dataidx_map = {}
            for site in sites:
                trainset = llm_Camelyon17(site=site, split='train', base_path=args.datadir,
                                      transform=transforms.ToTensor(),patch_h=64,patch_w=64)
                testset = llm_Camelyon17(site=site, split='test', base_path=args.datadir,
                                     transform=transforms.ToTensor(),patch_h=64,patch_w=64)
                val_len = math.floor(len(trainset) * 0.2)
                train_idx = list(range(len(trainset)))[:-val_len]
                val_idx = list(range(len(trainset)))[-val_len:]
                valset = torch.utils.data.Subset(trainset, val_idx)
                trainset = torch.utils.data.Subset(trainset, train_idx)
                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                # print(len(trainset))
        elif args.dataset == 'prostate':
            sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
            net_dataidx_map = {}
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            for site in sites:
                trainset = llm_Prostate(site=site, split='train', base_path=args.datadir, transform=transform,patch_h=64,patch_w=64)
                valset = llm_Prostate(site=site, split='val', base_path=args.datadir, transform=transform,patch_h=64,patch_w=64)
                testset = llm_Prostate(site=site, split='test', base_path=args.datadir, transform=transform,patch_h=64,patch_w=64)

                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                print(len(trainset))
        elif args.dataset == 'nuclei':
            
            args.imbalance = True
            
            sites = ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']
            net_dataidx_map = {}
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
            ])
            for site in sites:
                trainset = llm_Nuclei(site=site, split='train', base_path=args.datadir, transform=transform,patch_h=64,patch_w=64)
                valset = llm_Nuclei(site=site, split='val', base_path=args.datadir, transform=transform,patch_h=64,patch_w=64)
                testset = llm_Nuclei(site=site, split='test', base_path=args.datadir, transform=transform,patch_h=64,patch_w=64)
                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                print(len(trainset))

        min_data_len = min([len(s) for s in trainsets])
        for idx in range(len(trainsets)):
            if args.imbalance:
                trainset = trainsets[idx]
                valset = valsets[idx]
                testset = testsets[idx]
            else:
                trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len))))
                valset = valsets[idx]
                testset = testsets[idx]

            train_loaders.append(torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True))
            val_loaders.append(torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,pin_memory=True))
            test_loaders.append(torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,pin_memory=True))

    return sites, trainsets, testsets, train_loaders, val_loaders, test_loaders, net_dataidx_map

def recorded_multicenters_data_API_dataloader(args):
    '''the dataloader support for
    camelyon17/prostate/Nuclei/EC
    '''
    publish_data1=["camelyon17","prostate","nuclei"]
    private_data=["Endometrial_Cancer"]
    train_loaders, test_loaders = [], []
    val_loaders = []
    trainsets, testsets = [], []
    valsets = []
    sites=[]
    if args.dataset in publish_data1:
        if args.dataset == 'camelyon17':
            # args.lr = 1e-3
            # loss_fun = nn.CrossEntropyLoss()
            sites = ['1', '2', '3', '4', '5']
            net_dataidx_map = {}
            for site in sites:
                trainset = Camelyon17(site=site, split='train',base_path=args.datadir, transform=transforms.ToTensor())
                testset = Camelyon17(site=site, split='test',base_path=args.datadir, transform=transforms.ToTensor())
                val_len = math.floor(len(trainset) * 0.2)
                train_idx = list(range(len(trainset)))[:-val_len]
                val_idx = list(range(len(trainset)))[-val_len:]
                valset = torch.utils.data.Subset(trainset, val_idx)
                trainset = torch.utils.data.Subset(trainset, train_idx)
                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                # print(len(trainset))
        elif args.dataset == 'prostate':
            # args.lr = 1e-4
            # args.iters = 500
            # model = UNet(input_shape=[3, 384, 384])
            # loss_fun = JointLoss()
            sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
            net_dataidx_map = {}
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            for site in sites:
                trainset = Prostate(site=site, split='train',base_path=args.datadir, transform=transform)
                valset = Prostate(site=site, split='val',base_path=args.datadir, transform=transform)
                testset = Prostate(site=site, split='test',base_path=args.datadir, transform=transform)

                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                print(len(trainset))
        elif args.dataset == 'nuclei':
            # args.lr = 1e-4
            # args.iters = 500
            args.imbalance = True
            # model = UNet(input_shape=[3, 256, 256])
            # loss_fun = DiceLoss()
            sites = ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']
            net_dataidx_map = {}
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
            ])
            for site in sites:
                trainset = Nuclei(site=site, split='train',base_path=args.datadir, transform=transform)
                valset = Nuclei(site=site, split='val',base_path=args.datadir, transform=transform)
                testset = Nuclei(site=site, split='test',base_path=args.datadir, transform=transform)
                print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
                trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)
                net_dataidx_map[site] = trainset
                print(len(trainset))

        min_data_len = min([len(s) for s in trainsets])
        for idx in range(len(trainsets)):
            if args.imbalance:
                trainset = trainsets[idx]
                valset = valsets[idx]
                testset = testsets[idx]
            else:
                trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len))))
                valset = valsets[idx]
                testset = testsets[idx]

            train_loaders.append(torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,num_workers=4,pin_memory=True))
            val_loaders.append(torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,num_workers=4,pin_memory=True))
            test_loaders.append(torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,num_workers=4,pin_memory=True))
        return sites, trainsets, testsets, train_loaders, val_loaders, test_loaders,net_dataidx_map
    
    elif args.dataset in private_data:
        if args.dataset=="Endometrial_Cancer":
            net_dataidx_map = {}
            sites = ["A", "B", "C","D"]
            for site in sites:
                train_set = med_DataSet(root=args.datadir, center=site, data_set="traindata",
                                        transform=transforms.ToTensor())
                test_set = med_DataSet(root=args.datadir, center=site, data_set="testdata",
                                       transform=transforms.ToTensor())
                train_dl_local = data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,num_workers=4,pin_memory=True
                                                 )
                test_dl_local = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,num_workers=4,pin_memory=True)
                train_loaders.append(train_dl_local)
                test_loaders.append(test_dl_local)
                net_dataidx_map[site] = train_set
        return sites, trainsets, testsets, train_loaders, val_loaders, test_loaders,net_dataidx_map  # trainsets, testsets,val_loaders=[]
if __name__ == '__main__':
    exit()



