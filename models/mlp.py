import torch
import torch.nn as nn

class customMLP(nn.Module):
    def __init__(self, input_dim, hidden_feat, output_dim, activation, **kwargs):
        super().__init__()
        self.model = self.make_layer(input_dim, hidden_feat, output_dim, activation)
    def make_layer(self, input_dim, hidden_feat, output_dim, activation):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_feat),
            activation,
            nn.Linear(hidden_feat, output_dim),
        )
        return model
    def forward(self, x):
        return self.model(x)
    
        