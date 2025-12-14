import random
from typing import List

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(
        self, index
    ):
        self._index: List[dict] = index

    def __getitem__(self, ind):
        
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)
        data_label = data_dict["label"]
        instance_data = {"data_object": data_object, "labels": data_label}

        return instance_data

    def __len__(self):
        return len(self._index)

    def load_object(self, path):
        raise NotImplementedError("load_object() must be implemented in subclass")
