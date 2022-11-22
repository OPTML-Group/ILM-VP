import os
from PIL import Image
import six
import lmdb
import pickle
import json
from collections import OrderedDict
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import warnings

import sys
sys.path.append(".")
from cfg import *


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class LMDBDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__()
        db_path = os.path.join(root, f"{split}.lmdb")
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
        if split == 'train':
            try:
                with open(os.path.join(root, "example_dict.json"), "r") as f:
                    self.example_dict = json.load(f)
            except:
                warnings.warn("please add example dict first")

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf)

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    def get_class_example(self, class_id):
        img, _ = self.__getitem__(self.example_dict[str(class_id)])
        return img

class ImageNetCLSLMDBDataset(LMDBDataset):
    def __init__(self, root, split='train', class_id = 0, transform=None, target_transform=None):
        super().__init__(root, split, transform, target_transform)
        assert split == 'train'
        with open(os.path.join(root, f"{split}_class_split.json"), "r") as f:
            self.class_split_dict = json.load(f)
        self.class_split_dict["-1"] = -1
        self.class_id = int(class_id)

    def __getitem__(self, index):
        return super().__getitem__(1 + index + self.class_split_dict[str(self.class_id - 1)])

    def __len__(self):
        return self.class_split_dict[str(self.class_id)] - self.class_split_dict[str(self.class_id - 1)]


class COOPLMDBDataset(LMDBDataset):
    def __init__(self, root, split="train", transform=None) -> None:
        super().__init__(root, split, transform=transform)
        with open(os.path.join(root, "split.json")) as f:
            split_file = json.load(f)
        idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split_file["test"]}.items()))
        self.classes = list(idx_to_class.values())




def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


def dataset2lmdb(dataset, lmdb_path, write_frequency=5000):
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    isdir = os.path.isdir(lmdb_path)
    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
