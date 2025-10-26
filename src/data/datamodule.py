from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

class SVHNDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 256, num_workers: int = 4):
        super().__init__()
        self.data_dir, self.batch_size, self.num_workers = data_dir, batch_size, num_workers

    def setup(self, stage: Optional[str] = None):
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ])
        self.train = datasets.SVHN(self.data_dir, split="train", download=True, transform=tfm)
        self.test  = datasets.SVHN(self.data_dir, split="test",  download=True, transform=tfm)

    def train_dataloader(self): return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers, pin_memory=True)
    def val_dataloader(self):   return DataLoader(self.test,  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    def test_dataloader(self):  return self.val_dataloader()
