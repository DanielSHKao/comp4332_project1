import os
import pandas as pd
import numpy as np
import torch

import pytorch_lightning as pl

from torch.utils.data import Dataset
from utils.utils import load_data, tokenize, filter_stopwords, lower, Embedder

class CommentDataset(Dataset):
    def __init__(self, data_dir, columns=['text','stars'], split_name='train', embedder='en_core_web_sm', transform=None,**kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.columns = columns
        self.embedder = Embedder(embedder)
        self.transform = transform
        self.df = load_data(split_name, columns=self.columns, folder=self.data_dir)
        #self.df['tokens'] = self.df['text'].map(tokenize).map(filter_stopwords).map(lower)
        #self.df['vectors'] = self.df['text'].map(self.embedder.word_embedding)
        self.df['vectors'] = self.df['text'].map(self.embedder.sentence_embedding)


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        x = self.df['vectors'][index]
        y = self.df['stars'][index]
        if self.transform is not None:
            x = self.transform(x)
        #return x, y
        return torch.from_numpy(x), y-1
