import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as sw
import numpy as np
import spacy
import yaml
from argparse import Namespace
import csv
from ast import literal_eval
import nltk.data
def load_data(split_name='train', columns=['text', 'stars'], folder='./../data'):
    '''
        "split_name" may be set as 'train', 'valid' or 'test' to load the corresponding dataset.
        
        You may also specify the column names to load any columns in the .csv data file.
        Among many, "text" can be used as model input, and "stars" column is the labels (sentiment). 
        If you like, you are free to use columns other than "text" for prediction.
    '''
    try:
        print(f"select [{', '.join(columns)}] columns from the {split_name} split")
        df = pd.read_csv(f'{folder}/{split_name}.csv')
        df = df.loc[:,columns]
        print("Success")
        return df
    except:
        print(f"Failed loading specified columns... Returning all columns from the {split_name} split")
        df = pd.read_csv(f'{folder}/{split_name}.csv')
        return df
def load_preprocess(split_name='train', columns=['text', 'stars'], folder='./../data',embedding='sentence'):
    
    if embedding=='sentence':
        df=pd.read_csv(f'{folder}/{split_name}_{embedding}.csv',converters=dict(sent_embed=literal_eval))
        if split_name!="test":
            df = df.loc[:,columns+['sent_embed']]
    elif embedding=='word':
        print(f'Reading {split_name} csv file...')
        df=pd.read_csv(f'{folder}/{split_name}_{embedding}.csv')
        if split_name!="test":
            df = df.loc[:,columns+['word_embed']]
    elif embedding=='subtext':
        print(f'Reading {split_name} csv file...')
        df=pd.read_csv(f'{folder}/{split_name}_{embedding}.csv')
        if split_name!="test":
            df = df.loc[:,columns+['subtext_embed']]
    else:
        print(f'Reading {split_name} csv file...')
        df=pd.read_csv(f'{folder}/{split_name}_{embedding}.csv')
        if split_name!="test":
            df = df.loc[:,columns+['subsentence_embed']]
    return df
def lower(s):
    """
    :param s: a string.
    return a string with lower characters
    Note that we allow the input to be nested string of a list.
    e.g.
    Input: 'Text mining is to identify useful information.'
    Output: 'text mining is to identify useful information.'
    """
    if isinstance(s, list):
        return [lower(t) for t in s]
    if isinstance(s, str):
        return s.lower()
    else:
        raise NotImplementedError("unknown datatype")


def tokenize(text):
    """
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    e.g.
    Input: 'Text mining is to identify useful information.'
    Output: ['Text', 'mining', 'is', 'to', 'identify', 'useful', 'information', '.']
    """
    return nltk.word_tokenize(text)

def stem(tokens):
    """
    :param tokens: a list of tokens, type: list
    return a list of stemmed words, type: list
    e.g.
    Input: ['Text', 'mining', 'is', 'to', 'identify', 'useful', 'information', '.']
    Output: ['text', 'mine', 'is', 'to', 'identifi', 'use', 'inform', '.']
    """
    ### equivalent code
    # results = list()
    # for token in tokens:
    #     results.append(ps.stem(token))
    # return results
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]

def n_gram(tokens, n=1):
    """
    :param tokens: a list of tokens, type: list
    :param n: the corresponding n-gram, type: int
    return a list of n-gram tokens, type: list
    e.g.
    Input: ['text', 'mine', 'is', 'to', 'identifi', 'use', 'inform', '.'], 2
    Output: ['text mine', 'mine is', 'is to', 'to identifi', 'identifi use', 'use inform', 'inform .']
    """
    if n == 1:
        return tokens
    else:
        results = list()
        for i in range(len(tokens)-n+1):
            # tokens[i:i+n] will return a sublist from i th to i+n th (i+n th is not included)
            results.append(" ".join(tokens[i:i+n]))
        return results

def filter_stopwords(tokens):
    """
    :param tokens: a list of tokens, type: list
    return a list of filtered tokens, type: list
    e.g.
    Input: ['text', 'mine', 'is', 'to', 'identifi', 'use', 'inform', '.']
    Output: ['text', 'mine', 'identifi', 'use', 'inform', '.']
    """
    ### equivalent code
    # results = list()
    # for token in tokens:
    #     if token not in stopwords and not token.isnumeric():
    #         results.append(token)
    # return results
    stopwords = set(sw.words('english'))
    return [token for token in tokens if token not in stopwords and not token.isnumeric()]

def get_onehot_vector(feats, feats_dict):
    """
    :param data: a list of features, type: list
    :param feats_dict: a dict from features to indices, type: dict
    return a feature vector,
    """
    # initialize the vector as all zeros
    vector = np.zeros(len(feats_dict), dtype=np.float)
    for f in feats:
        # get the feature index, return -1 if the feature is not existed
        f_idx = feats_dict.get(f, -1)
        if f_idx != -1:
            # set the corresponding element as 1
            vector[f_idx] = 1
    return vector

class Embedder:
    def __init__(self,embedder,tokenizer):
        self.embedder=spacy.load(embedder)
        self.tokenizer=tokenizer
    def word_embedding(self,text):
        tokens = lower(filter_stopwords(tokenize(text)))
        result = np.zeros((128,96))
        i=0
        for token in tokens:
            if i<64:
                result[i]=(self.embedder(token).vector.tolist())
            else:
                break
            i+=1
        return result.tolist()
    def sentence_embedding(self,text):
        return self.embedder(text).vector.tolist()
    def subsentence_embedding(self,text):
        tokens = tokenize(text)
        sublen = len(tokens)//10
        subsentences = [' '.join(tokens[i*sublen:(i+1)*sublen]) for i in range(9)]
        subsentences.append(' '.join(tokens[(9)*sublen:]))
        result = np.zeros((10,96))
        for i in range(10):
            v=(self.embedder(subsentences[i]).vector.tolist())
            if len(v)==0:
                v=[0]*96
            result[i]= v
        return result.tolist()
    def subtext_embedding(self,text):
        sentences = self.tokenizer.tokenize(text)
        result = np.zeros((10,96))
        i=0
        for s in sentences:
            if i<10:
                result[i]=(self.embedder(s).vector.tolist())
            else:
                break
            i+=1
        return result.tolist()


def load_cfg(cfg):
    hyp = None
    if isinstance(cfg, str):
        with open(cfg, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    return Namespace(**hyp)

def merge_args_cfg(args, cfg):
    dict0 = vars(args)
    dict1 = vars(cfg)
    dict = {**dict0, **dict1}

    return Namespace(**dict)

def write_to_csv(file,df):
    df.to_csv(file)

