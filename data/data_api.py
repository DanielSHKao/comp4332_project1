import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils.utils import *
from data.data import CommentDataset
import random

class CommentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, columns = ['text','stars'],**kwargs):
        self.num_workers=num_workers
        self.batch_size=batch_size
        self.train_set = CommentDataset(data_dir,columns = columns, split_name = 'train', kwargs)
        self.val_set = CommentDataset(data_dir,columns = columns, split_name = 'valid', kwargs)
        self.test_set = CommentDataset(data_dir,columns = columns, split_name = 'test', kwargs)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle= True, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle= True, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
