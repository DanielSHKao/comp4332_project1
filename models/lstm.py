import torch
import torch.nn as nn
class customLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size)
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        x, _ = self.lstm(x.view(len(x), 1, -1))
        x = torch.flatten(x,dim=1)
        logits = self.fc(x)
        return logits
    