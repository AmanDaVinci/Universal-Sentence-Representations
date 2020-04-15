import torch
import torch.nn as nn

class Classifier(nn.Module):
    
    def __init__(self, encoder, encoded_dim):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.Sequential(
            nn.Linear(4 * encoded_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
    
    def forward(self, premise, hypothesis):
        u = self.encoder(premise)
        v = self.encoder(hypothesis)
        diff = u - v
        prod = u * v
        x = torch.cat((u, v, diff, prod), dim=1)
        return self.layers(x)