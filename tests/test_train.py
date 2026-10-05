import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from surrogate.model import MLP
from surrogate.train import test_loop as evaluate
from surrogate.train import train_loop

DEVICE = torch.device("cpu")


@pytest.fixture
def linear_problem():
    gen = torch.Generator().manual_seed(0)
    X = torch.randn(64, 4, generator=gen)
    W = torch.randn(4, 3, generator=gen)
    return X, X @ W


def test_train_loop_decreases_loss(linear_problem):
    torch.manual_seed(0)
    X, y = linear_problem
    dl = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    model = MLP(4, 3, hidden=(32,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    first = train_loop(dl, model, optimizer, DEVICE)
    for _ in range(100):
        last = train_loop(dl, model, optimizer, DEVICE)
    assert last < 0.05 * first


def test_eval_does_not_change_weights(linear_problem):
    X, y = linear_problem
    dl = DataLoader(TensorDataset(X, y), batch_size=16)
    model = MLP(4, 3, hidden=(8,))
    before = [p.clone() for p in model.parameters()]
    evaluate(dl, model, DEVICE)
    for p, q in zip(model.parameters(), before):
        assert torch.equal(p, q)


def test_eval_returns_dataset_mse_with_uneven_batches(linear_problem):
    X, y = linear_problem
    model = MLP(4, 3, hidden=(8,))
    dl = DataLoader(TensorDataset(X, y), batch_size=7)  # 64 = 9 x 7 + 1
    with torch.no_grad():
        expected = torch.nn.functional.mse_loss(model(X), y).item()
    assert evaluate(dl, model, DEVICE) == pytest.approx(expected, rel=1e-5)
