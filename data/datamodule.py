# data/datamodule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import ImageDataset

class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        train_tfms,
        val_tfms,
        batch_size,
        num_workers,
        train_sampler=None,
    ):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_tfms = train_tfms
        self.val_tfms = val_tfms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sampler = train_sampler

    def setup(self, stage=None):
        self.train_ds = ImageDataset(self.train_df, self.train_tfms)
        self.val_ds = ImageDataset(self.val_df, self.val_tfms)
        self.test_ds = ImageDataset(self.test_df, self.val_tfms)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=self.train_sampler,  # ← ВАЖНО
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        
