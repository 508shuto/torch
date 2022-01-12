import torch.nn as nn
import torch

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, ypred, y):
        ypred.float()
        y.float()
        return torch.sqrt(self.mse(ypred, y)).float()