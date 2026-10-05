import pytest
import torch
from torch import nn

from surrogate.model import MLP


def test_output_shape():
    model = MLP(4, 50, hidden=(16, 16))
    assert model(torch.randn(8, 4)).shape == (8, 50)


def test_each_hidden_layer_has_its_own_activation():
    model = MLP(4, 10, hidden=(8, 8, 8), activations=nn.PReLU)
    activations = [m for m in model.net if isinstance(m, nn.PReLU)]
    assert len(activations) == 3
    assert len({id(a) for a in activations}) == 3


def test_last_layer_is_linear_without_activation():
    model = MLP(4, 10, hidden=(8, 8))
    assert isinstance(model.net[-1], nn.Linear)
    assert model.net[-1].out_features == 10


@pytest.mark.parametrize("hidden", [(32,), (64, 32, 16)])
def test_layer_sizes_chain(hidden):
    model = MLP(3, 7, hidden=hidden)
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    sizes = [linears[0].in_features] + [layer.out_features for layer in linears]
    assert sizes == [3, *hidden, 7]
