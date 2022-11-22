import os
import json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from .dataset_lmdb import COOPLMDBDataset, LMDBDataset, ImageNetCLSLMDBDataset
from .abide import ABIDE
from .const import GTSRB_LABEL_MAP, IMAGENETCLASSES, IMAGENETNORMALIZE

def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names

def get_class_names_from_split(root):
    with open(os.path.join(root, "split.json")) as f:
        split = json.load(f)["test"]
    idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split}.items()))
    return list(idx_to_class.values())

def prepare_ar_resnet_data(dataset, data_path):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "cifar100":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "gtsrb":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "svhn":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.SVHN(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': [f'{i}' for i in range(10)],
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "abide":
        preprocess = transforms.ToTensor()
        D = ABIDE(root = data_path)
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': ["non ASD", "ASD"],
            'mask': D.get_mask(),
        }
    elif dataset in ["food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")
    return loaders, configs


def prepare_clip_data(dataset, data_path, preprocess):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = preprocess)
        class_names = [f'{i}' for i in range(10)]
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "abide":         
        D = ABIDE(root = data_path)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        class_names = ["non ASD", "ASD"]
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names


def prepare_imagenet_data(data_path, shuffle_train=True):
    data_path = os.path.join(data_path, "imagenet")
    batch_size = 256
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    train_data = LMDBDataset(root = data_path, split='train', transform=preprocess)
    test_data = LMDBDataset(root = data_path, split='val', transform=preprocess)
    loaders = {
        'train': DataLoader(train_data, batch_size, shuffle = shuffle_train, num_workers=8),
        'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=8),
    }
    configs = {
        'class_names': refine_classnames(IMAGENETCLASSES),
        'batch_size': batch_size,
    }
    return loaders, configs


def prepare_imagenet_classwise_data(data_path, class_id):
    data_path = os.path.join(data_path, "imagenet")
    batch_size = 2000
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    train_data = ImageNetCLSLMDBDataset(root = data_path, split='train', class_id=class_id, transform=preprocess)
    train_loader = DataLoader(train_data, batch_size, shuffle = False, num_workers=8)
    x, _ = next(iter(train_loader))
    return x


def prepare_gtsrb_fraction_data(data_path, fraction, preprocess=None):
    data_path = os.path.join(data_path, "gtsrb")
    assert 0 < fraction <= 1
    new_length = int(fraction*26640)
    indices = torch.randperm(26640)[:new_length]
    sampler = SubsetRandomSampler(indices)
    if preprocess == None:
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
        return loaders, configs
    else:
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        return loaders, class_names

def prepare_data_resolution(dataset, data_path, size):
    if dataset == "cifar10":
        data_path = os.path.join(data_path, "cifar10")
        preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((size, size)),
        }
    elif dataset in ["food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        data_path = os.path.join(data_path, dataset)
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((size, size)),
        }
    return loaders, configs