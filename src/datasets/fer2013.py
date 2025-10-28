import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    def __init__(self, csv_path, usage="Training", transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Usage"] == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
        img = pixels.reshape(48, 48)
        im = Image.fromarray(img, mode="L").convert("RGB")
        y = int(row["emotion"]) 
        if self.transform:
            im = self.transform(im)
        return im, y
