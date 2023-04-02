import os
import pandas as pd
import numpy as np

import pytorch_lightning as pl

from torch.utils.data import Dataset
from utils.utils import load_data, tokenize, filter_stopwords, lower

class CommentDataset(Dataset):
    def __init__(self, data_dir, columns = ['text','stars'], split_name = 'train', **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.columns = columns
        self.df = load_data(split_name, columns=columns, folder=data_dir)
        self.df['tokens'] = self.df['text'].map(tokenize).map(filter_stopwords).map(lower)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return dict(self.df.iloc[index])