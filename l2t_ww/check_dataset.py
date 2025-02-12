import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data


class FolderSubset(data.Dataset):
    def __init__(self, dataset, classes, indices):
        self.dataset = dataset
        self.classes = classes
        #print(self.classes)
        self.indices = indices

        self.update_classes()

    def update_classes(self):
        for i in self.indices:
            img_path, cls = self.dataset.samples[i]
            #print(img_path,cls)
            cls = self.classes.index(cls)
            #print(img_path, cls)
            self.dataset.samples[i] = (img_path, cls)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class STL10Subset(data.Dataset):
    def __init__(self, dataset, classes, indices):
        self.dataset = dataset
        self.classes = classes
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class CIFARSubset(data.Dataset):
    def __init__(self, dataset, classes, indices):
        self.dataset = dataset
        self.classes = classes
        self.indices = indices

        # self.update_classes()

    # def update_classes(self):
    #     for i in self.indices:
    #         if self.dataset.train:
    #             self.dataset.train_labels[i] = self.classes.index(self.dataset.train_labels[i])
    #         else:
    #             self.dataset.test_labels[i] = self.classes.index(self.dataset.test_labels[i])

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def check_split(opt,isnodules=False,nodules_path=r'G:\L2T-ww-master_transformer1\lung_nodules'):
    splits = []

    if isnodules:

        for split in ['train_data','train_data','test_data']:
            label=[]
            counter = 0
            img_index=[]
            counter1=0
            for sample in os.listdir(os.path.join(nodules_path,split)):
                label.append(counter)
                counter=counter+1
                for img in os.listdir(os.path.join(nodules_path,split,sample)):
                    t1=img.split("_")[-1]
                    t2=t1.split(".")[0]
                    img_index.append(eval(t2))

            splits.append((label,img_index))
    else:
        for split in ['train', 'val', 'test']:
            splits.append(torch.load('split/' + opt.datasplit + '-' + split))
    return splits


def check_dataset(opt):
    normalize_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406),
                                                                   (0.229, 0.224, 0.225))])
    train_large_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip()])
    val_large_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224)])
    train_small_transform = transforms.Compose([transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip()])

    # splits = check_split(opt)

    if opt.dataset in ['cub200', 'indoor', 'stanford40', 'dog']:
        splits = check_split(opt)
        train, val = 'train', 'test'
        train_transform = transforms.Compose([train_large_transform, normalize_transform])
        val_transform = transforms.Compose([val_large_transform, normalize_transform])
        sets = [dset.ImageFolder(root=os.path.join(opt.dataroot, train), transform=train_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, train), transform=val_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, val), transform=val_transform)]
        sets = [FolderSubset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])

    elif opt.dataset == 'stl10':
        splits = check_split(opt)
        train_transform = transforms.Compose([transforms.Resize(32),
                                              train_small_transform, normalize_transform])
        val_transform = transforms.Compose([transforms.Resize(32), normalize_transform])
        sets = [dset.STL10(opt.dataroot, split='train', transform=train_transform, download=True),
                dset.STL10(opt.dataroot, split='train', transform=val_transform, download=True),
                dset.STL10(opt.dataroot, split='test', transform=val_transform, download=True)]
        sets = [STL10Subset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])

    elif opt.dataset in ['cifar10', 'cifar100']:
        splits = check_split(opt)
        train_transform = transforms.Compose([train_small_transform, normalize_transform])
        val_transform = normalize_transform
        CIFAR = dset.CIFAR10 if opt.dataset == 'cifar10' else dset.CIFAR100

        sets = [CIFAR(opt.dataroot, download=True, train=True, transform=train_transform),
                CIFAR(opt.dataroot, download=True, train=True, transform=val_transform),
                CIFAR(opt.dataroot, download=True, train=False, transform=val_transform)]
        sets = [CIFARSubset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])
    elif opt.dataset in ['lung_nodules']:
        splits = check_split(opt,isnodules=True)
        train, val = 'train_data', 'test_data'
        train_transform = transforms.Compose([train_large_transform, normalize_transform])
        val_transform = transforms.Compose([val_large_transform, normalize_transform])
        sets = [dset.ImageFolder(root=os.path.join(opt.dataroot, train), transform=train_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, train), transform=val_transform),
                dset.ImageFolder(root=os.path.join(opt.dataroot, val), transform=val_transform)]
        sets = [FolderSubset(dataset, *split) for dataset, split in zip(sets, splits)]

        opt.num_classes = len(splits[0][0])
    else:
        raise Exception('Unknown dataset')

    loaders = [torch.utils.data.DataLoader(dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=0) for dataset in sets]
    return loaders
