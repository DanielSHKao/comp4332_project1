import torch
import torch.nn as nn
class customLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,depth, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size, batch_first=True, num_layers=depth)
        self.fc = nn.Linear(hidden_size,num_classes)
        self.flatten=nn.Flatten()
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

class biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth,max_length, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size, batch_first=True, num_layers=depth,bidirectional=True)
        self.fc = nn.Linear(hidden_size*max_length*2,num_classes)
        self.flatten=nn.Flatten()
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits
class m2mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,depth, max_length,**kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size, batch_first=True, num_layers=depth)
        self.fc = nn.Linear(hidden_size*max_length,num_classes)
        self.flatten=nn.Flatten()
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits