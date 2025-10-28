import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DAiSEEDataset(Dataset):
    def __init__(self, manifest_csv, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im = Image.open(row["path"]).convert("RGB")
        e = int(row["engagement"]) 
        f = int(row["frustration"]) 
        if self.transform:
            im = self.transform(im)
        return im, e, f
