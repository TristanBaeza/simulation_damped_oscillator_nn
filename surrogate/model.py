from torch import nn


class MLP(nn.Module):
    def __init__(self, dimension_in, dimension_out, hidden, activations=nn.GELU):
        super().__init__()
        layers, dimension = [], dimension_in
        for h in hidden:
            activation = activations()  # new instance per layer: no shared weights
            layers += [nn.Linear(dimension, h), activation]
            dimension = h
        layers.append(nn.Linear(dimension, dimension_out))  # no activation: regression
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
