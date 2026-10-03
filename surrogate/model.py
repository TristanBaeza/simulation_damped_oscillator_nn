from torch import nn
import torch


class MLP(nn.Module):
    def __init__(
        self, dimension_in, dimension_out, hidden=(128, 128, 128), activations=nn.GELU
    ):
        super().__init__()
        layers, dimension = [], dimension_in
        for h in hidden:
            activation = activations()
            layers += [nn.Linear(dimension, h), activation]
            dimension = h
        layers.append(nn.Linear(dimension, dimension_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
