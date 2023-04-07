import os
import pandas as pd
import numpy as np
import torch

import pytorch_lightning as pl

from torch.utils.data import Dataset
from utils.utils import *
from ast import literal_eval
class CommentDataset(Dataset):
    def __init__(self, data_dir, columns=['text','stars'], split_name='train', embedder='en_core_web_sm', embedding='sentence', transform=None,**kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.columns = columns
        embedder = Embedder(embedder)
        self.transform = transform
        if embedding=='sentence':
            self.key = 'sent_embed'
            embedding_method = embedder.sentence_embedding
        elif embedding=='word':
            self.key = 'word_embed'
            embedding_method = embedder.word_embedding
        else:
            self.key = 'sub_embed'
            embedding_method = embedder.subsentence_embedding

        try:
            self.df = load_preprocess(split_name,columns=columns,folder=data_dir,embedding=embedding)
            print(f"Use pre-process embedding vectors for {split_name} dataset.")
            self.vectors = self.df[self.key]
        except:
            self.df = load_data(split_name, columns=columns, folder=data_dir)
            print(f"Embedding {split_name} dataset...")
            self.df[self.key] = self.df['text'].map(embedding_method)
            self.vectors = self.df[self.key]

            write_to_csv(f'{data_dir}/{split_name}_{embedding}.csv',self.df)
            print(f"Done")
        #self.df['tokens'] = self.df['text'].map(tokenize).map(filter_stopwords).map(lower)
        #self.df['vectors'] = self.df['text'].map(self.embedder.word_embedding)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        x = self.vectors[index]
        if type(x)==str:
            x = literal_eval(x)
        y = self.df['stars'][index]
        if self.transform is not None:
            x = self.transform(x)
        #return x, y-1
        return torch.tensor(x).float(), y-1 