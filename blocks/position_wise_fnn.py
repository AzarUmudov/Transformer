import torch
import torch.nn as nn
import math 

class PositionWiseFNN(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff 
        self.dropout = nn.Dropout(p=dropout)
        self.layer1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.layer2 = nn.Linear(in_features=d_ff, out_features=d_model)
    
    def forward(self, x):
        x = nn.Relu(self.layer1(x))
        x = self.layer2(self.dropout(x))
        return x

