import os
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from data.data import CommentDataset
import random

class CommentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, columns, batch_size, num_workers, embedder, embedding, max_length,**kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.columns = columns
        self.embedder = embedder
        self.embedding = embedding
        self.max_length = max_length
    def setup(self, stage: str):
        self.train_set = CommentDataset(self.data_dir, columns=self.columns, split_name='train', embedder= self.embedder,embedding=self.embedding,max_length=self.max_length)
        self.val_set = CommentDataset(self.data_dir, columns=self.columns, split_name='valid', embedder= self.embedder,embedding=self.embedding,max_length=self.max_length)
        self.test_set = CommentDataset(self.data_dir, columns=['all'], split_name='test', embedder= self.embedder,embedding=self.embedding,max_length=self.max_length)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
