from tqdm import tqdm
import json
import numpy as np

import sys
sys.path.append(".")
from cfg import *
from datatools.dataset_lmdb import LMDBDataset

if __name__ == "__main__":
    for dataset in ["imagenet"]:
        data_path = os.path.join(data_path, dataset)
        for split in ["train", "test"]:
            data = LMDBDataset(data_path, split=split)
            res_dict = {0: 0}
            for i in tqdm(range(len(data)), ncols=100):
                y = data.__getitem__(i)[1]
                assert y >= np.array(list(res_dict.keys())).max()
                res_dict[int(y)] = i
            with open(os.path.join(data_path, f"{split}_class_split.json"), "w") as f:
                json.dump(res_dict, f)