import random
from typing import List

import torch
from src.datasets.base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
from src.transforms.transforms import base_transform


class AircraftDataset(BaseDataset):
    def __init__(self, data_dir=None, num_folders=5, *args, **kwargs):
        self.data_dir = data_dir
        self.num_folders = num_folders
        index= self._create_index()

        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        index = []
        for num in range(self.num_folders):
            folder_path = Path(f'{self.data_dir}/data/data_{num}')
            print(folder_path)

            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    index.append({
                        'path': str(file_path),
                        'label': 'aircraft'
                    })

        return index

    def load_object(self, path):
        image = Image.open(path).convert("RGB")
        return image

    def __getitem__(self, idx, transform=base_transform):
        sample = self._index[idx]
        image = Image.open(sample["path"]).convert("RGB")
        label = sample["label"]

        if transform:
            image = transform(image)

        return {"data_object": image, "labels": label}