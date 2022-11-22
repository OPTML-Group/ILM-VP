import os
import json
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset


class COOPDataset(Dataset):
    def __init__(self, root, split="train", transform=None, loader='image') -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        with open(os.path.join(self.root, "split.json")) as f:
            split_file = json.load(f)
        self.split = split_file[split]
        idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split_file["test"]}.items()))
        self.classes = list(idx_to_class.values())
        self.loader = loader

    def __getitem__(self, index: int):
        path, label, _ = self.split[index]
        path = os.path.join(self.root, "images", path)
        if self.loader == 'image':
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
        elif self.loader == 'raw':
            with open(path, 'rb') as f:
                img = f.read()
        return img, label

    def __len__(self) -> int:
        return len(self.split)

