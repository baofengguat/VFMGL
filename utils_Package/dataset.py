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
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.datasets import ImageFolder, DatasetFolder
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

class Camelyon17(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'test']
        assert int(site) in [1,2,3,4,5] # five hospital

        base_path = base_path if base_path is not None else '.\camelyon17'
        self.base_path = base_path

        data_dict = np.load('./camelyon17/data.pkl', allow_pickle=True)
        self.paths, self.labels = data_dict[f'hospital{site}'][f'{split}']

        self.transform = transform
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image=image.resize((224,224))
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class Prostate(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'BIDMC':3, 'HK':3, 'I2CVB':3, 'ISBI':3, 'ISBI_1.5':3, 'UCL':3}
        assert site in list(channels.keys())
        self.split = split
        
        base_path = base_path if base_path is not None else '../data/prostate'
        
        images, labels = [], []
        sitedir = os.path.join(base_path, site)

        ossitedir = np.load("../data/prostate/{}-dir.npy".format(site)).tolist()

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
                    image = np.array(image_v[i-1:i+2, :, :])
                    image = np.transpose(image,(1,2,0))
                    
                    labels.append(label)
                    images.append(image)
        labels = np.array(labels).astype(int)
        images = np.array(images)

        index = np.load("../data/prostate/{}-index.npy".format(site)).tolist()

        labels = labels[index]
        images = images[index]

        trainlen = 0.8 * len(labels) * 0.8
        vallen = 0.8 * len(labels) - trainlen
        testlen = 0.2 * len(labels)

        if(split=='train'):
            self.images, self.labels = images[:int(trainlen)], labels[:int(trainlen)]

        elif(split=='val'):
            self.images, self.labels = images[int(trainlen):int(trainlen + vallen)], labels[int(trainlen):int(trainlen + vallen)]
        else:
            self.images, self.labels = images[int(trainlen + vallen):], labels[int(trainlen + vallen):]

        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.long).squeeze()

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

            image = np.transpose(image,(2, 0, 1))
            image = torch.Tensor(image)
            
            label = self.transform(label)

        return image, label


class Nuclei(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'val', 'test']
        assert site in ['SiteA', 'SiteB', 'SiteC', 'SiteD', 'SiteE', 'SiteF']
        
        self.base_path = base_path if base_path is not None else r'./NucSeg'
        self.base_path = os.path.join(self.base_path, site)

        images = []
        labels = []

        if split == 'train' or split == 'val':
            self.base_path = os.path.join(self.base_path, split)
            img_path = os.path.join(self.base_path, "images")
            lbl_path = os.path.join(self.base_path, "labels")
            for i in os.listdir(img_path):
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
        image = Image.open(self.images[idx].replace("\\","/")).convert('RGB')



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

        return image, label

def get_lung_leison_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0,center_name="A"):
    if dataset=="lung_nodules":
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir + '/%s/train_data/'%center_name, dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir + '/%s/test_data/'%center_name, transform=transform_test)
        print(f'[Client {center_name}] Train={len(train_ds)}, Test={len(test_ds)}')
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True,num_workers=8,pin_memory=True,prefetch_factor=4)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False,num_workers=8,pin_memory=True,prefetch_factor=4)
    else:
        raise
    return train_dl, test_dl, train_ds, test_ds

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

        return  img, mask

if __name__=='__main__':
    exit()



