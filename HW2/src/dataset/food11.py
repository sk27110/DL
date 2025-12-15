import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.dataset.base_dataset import BaseDataset

class FoodDataset(BaseDataset):
    def __init__(self, data_path, mode="train", transforms=None, *args, **kwargs):
        self.path = data_path
        self.mode = mode
        self.transforms = transforms
        self.data_class = {
            "apple_pie": 0,
            "cheesecake": 1,
            "chicken_curry": 2,
            "french_fries": 3,
            "fried_rice": 4,
            "hamburger": 5,
            "hot_dog": 6,
            "ice_cream": 7,
            "omlette": 8,
            "pizza": 9,
            "sushi": 10
        }

        if mode == "train":
            self.part_path = Path(data_path)/"food11/train"
        else:
            self.part_path = Path(data_path)/"food11/test"
        
        index = self._create_index()
        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        path = self.part_path
        index = []

        for class_name, class_label in tqdm(self.data_class.items(), desc = "Indexing"):
            class_path = Path(self.part_path/class_name)
            if not class_path.exists():
                continue
            for img_path in class_path.glob("*"): 
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    # index.append({img_path, class_label})
                    index.append({
                    "path": str(img_path.resolve()),
                    "label": class_label
                    })
        return index
    

    def __getitem__(self, ind):
        sample = self._index[ind]

        image = Image.open(sample["path"]).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        return {
            "image": image,
            "label": sample["label"]
        }


