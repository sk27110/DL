import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.dataset.base_dataset import BaseDataset


class GtsrbDataset(BaseDataset):
    def __init__(self, data_path=None, mode="train", transforms=None, *args, **kwargs):
        """
        Args:
            data_path (str|Path): путь к корневой папке с изображениями (обычно .../versions/1/)
            csv_file (str|Path): путь к Train.csv или Test.csv
            mode (str): 'train' или 'test'
            transforms: torchvision.transforms для аугментаций и преобразований
        """
        self.data_path = Path(data_path)
        self.transforms = transforms
        self.mode=mode
        if mode == "test":
            self.csv_file = Path(data_path)/"Test.csv"
        else:
            self.csv_file = Path(data_path)/"Train.csv"

        index = self._create_gtsrb_index()
        super().__init__(index, *args, **kwargs)

    def _create_gtsrb_index(self):
        """
        Читает CSV и формирует индекс
        [
          {
            "path": ".../Train/20/00020_00000_00000.png",
            "label": 20,
            "width": 27,
            "height": 26,
            "bbox": [x1,y1,x2,y2]
          },
          ...
        ]
        """
        df = pd.read_csv(self.csv_file)
        index = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Indexing {self.mode}"):
            img_rel_path = row["Path"]   
            img_path = self.data_path / img_rel_path

            index.append({
                "path": str(img_path.resolve()),
                "label": int(row["ClassId"])
            })
        return index

    def __getitem__(self, idx):
        sample = self._index[idx]

        image = Image.open(sample["path"]).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        return {
            "image": image,
            "label": sample["label"]
        }

    def __len__(self):
        return len(self._index)
