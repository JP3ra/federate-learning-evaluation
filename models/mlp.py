import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, 1, 28, 28] or [batch_size, 784]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
