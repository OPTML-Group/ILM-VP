import h5py
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class ABIDE(Dataset):
    def __init__(self, root, transform=None) -> None:
        super().__init__()
        self.transform = transform
        data = []
        targets = []
        excel_file = pd.read_csv(os.path.join(root, "Phenotypic_V1_0b_preprocessed1.csv"))
        with h5py.File(os.path.join(root, "abide.hdf5"), 'r') as f:
            for key in f['patients'].keys():
                data.append(
                    np.expand_dims(self.data_array_to_matrix(
                        self.norm(np.array(f['patients'][key]['cc200']))
                        ), -1)
                    )
                targets.append(2 - excel_file[excel_file["FILE_ID"]==key]["DX_GROUP"])
        data = np.stack(data, axis=0)
        self.data = np.repeat(data, 3, -1).astype(np.float32)
        self.targets = np.concatenate(targets, axis=0)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]

    @staticmethod
    def data_array_to_matrix(data):
        M = np.zeros((200, 200))
        idx = 0
        for j in range(0,200):
            k = 0
            while k < 200:
                if k <= j: k += 1
                else:
                    if idx < data.shape[0]:
                        M[j][k] = data[idx]
                        idx += 1
                        k += 1
                    else: break
        return M
    
    @staticmethod
    def norm(x):
        x -= np.min(x)
        x /= np.max(x)
        return x

    @staticmethod
    def get_mask():
        central_size=200
        idx = 0
        M = np.ones((central_size, central_size))
        for j in range(0, central_size):
            k = 0
            while k < 200:
                if k <= j: k += 1
                else:
                    if idx < 19900:
                        M[j][k] = 0
                        idx += 1
                        k += 1
                    else: break
        return M