import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import torch.nn as nn
from timm.models.registry import register_model
from models.lstm import *
from models.mlp import *
"""######################################################################################################"""

@register_model
def lstm_256(input_size,num_classes,**kwargs):
    return customLSTM(input_size=input_size,hidden_size=256,num_classes=num_classes)

@register_model
def mlp_1k(input_size,num_classes, **kwargs):
    return customMLP(input_dim=input_size,hidden_feat=1024, output_dim=num_classes, activation=nn.ReLU())

@register_model
def dnn4_256(input_size,num_classes, **kwargs):
    return customDNN(input_dim=input_size,hidden_feat=256, output_dim=num_classes, activation=nn.ReLU())

@register_model
def lstm3_256(input_size,num_classes, max_length,**kwargs):
    return customLSTM(input_size=input_size, hidden_size=256, num_classes=num_classes,depth=2)

@register_model
def lstm2_256(input_size,num_classes, max_length,**kwargs):
    return customLSTM(input_size=input_size, hidden_size=256, num_classes=num_classes,depth=1)
@register_model
def lstm2_512(input_size,num_classes, max_length,**kwargs):
    return customLSTM(input_size=input_size, hidden_size=512, num_classes=num_classes,depth=1)
@register_model
def lstm3_1k(input_size,num_classes, max_length,**kwargs):
    return customLSTM(input_size=input_size, hidden_size=1024, num_classes=num_classes,depth=2)

@register_model
def lstm2_1k(input_size,num_classes, max_length,**kwargs):
    return customLSTM(input_size=input_size, hidden_size=1024, num_classes=num_classes,depth=1)

@register_model
def m2mlstm3_256(input_size,num_classes,max_length, **kwargs):
    return m2mLSTM(input_size=input_size, hidden_size=256, num_classes=num_classes,depth=2,max_length=max_length)

@register_model
def m2mlstm3_512(input_size,num_classes, max_length,**kwargs):
    return m2mLSTM(input_size=input_size, hidden_size=512, num_classes=num_classes,depth=2,max_length=max_length)

@register_model
def m2mlstm2_512(input_size,num_classes, max_length,**kwargs):
    return m2mLSTM(input_size=input_size, hidden_size=512, num_classes=num_classes,depth=1,max_length=max_length)

@register_model
def bilstm3_256(input_size,num_classes,max_length, **kwargs):
    return biLSTM(input_size=input_size, hidden_size=256, num_classes=num_classes,depth=2,max_length=max_length)

@register_model
def bilstm2_256(input_size,num_classes, max_length,**kwargs):
    return biLSTM(input_size=input_size, hidden_size=256, num_classes=num_classes,depth=1,max_length=max_length)