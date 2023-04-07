import torch
import torch.nn as nn
class customLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size, batch_first=True, num_layers=2,dropout=0.1)
        self.fc = nn.Linear(hidden_size,num_classes)
        self.flatten=nn.Flatten()
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.flatten(x)
        logits = self.fc(x)
        return logits
    