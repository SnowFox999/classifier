# data/dataset.py

from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, "path"]).convert("RGB")
        label = torch.tensor(self.df.loc[idx, "label"], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label
