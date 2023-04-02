import os
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from data.data import CommentDataset

class CommentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, columns=['text','stars'], **kwargs):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_set = CommentDataset(data_dir, columns=columns, split_name='train')
        self.val_set = CommentDataset(data_dir, columns=columns, split_name='valid')
        self.test_set = CommentDataset(data_dir, columns=columns, split_name='test')
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)