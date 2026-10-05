import os

import pytest
import torch

from conftest import ROOT, requires_data
from surrogate.data import loader

pytestmark = requires_data


@pytest.fixture(scope="module")
def loaded():
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        return loader()
    finally:
        os.chdir(cwd)


def tensors(dataloader):
    return dataloader.dataset.tensors


def test_split_sizes(loaded):
    train_loader, val_loader, test_loader, *_ = loaded
    n_train, n_val, n_test = (
        len(dl.dataset) for dl in (train_loader, val_loader, test_loader)
    )
    n = n_train + n_val + n_test
    assert n_train == int(0.8 * n)
    assert n_val == int(0.1 * n)


def test_splits_are_disjoint(loaded):
    # x0 vient d'un linspace : chaque simulation a une valeur unique
    x0 = [tensors(dl)[0][:, 0] for dl in loaded[:3]]
    assert not torch.isin(x0[0], x0[1]).any()
    assert not torch.isin(x0[0], x0[2]).any()
    assert not torch.isin(x0[1], x0[2]).any()


def test_batch_shapes_and_dtype(loaded):
    train_loader = loaded[0]
    X, y = next(iter(train_loader))
    assert X.shape == (train_loader.batch_size, 4)
    assert y.ndim == 2
    assert X.dtype == y.dtype == torch.float32


def test_no_nan(loaded):
    for dl in loaded[:3]:
        X, y = tensors(dl)
        assert not X.isnan().any()
        assert not y.isnan().any()


def test_train_inputs_are_standardized(loaded):
    X, _ = tensors(loaded[0])
    torch.testing.assert_close(X.mean(dim=0), torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(X.std(dim=0), torch.ones(4), atol=1e-5, rtol=0)


def test_train_outputs_are_standardized(loaded):
    _, y = tensors(loaded[0])
    assert y.mean().item() == pytest.approx(0, abs=1e-5)
    assert y.std().item() == pytest.approx(1, abs=1e-5)


def test_normalization_stats_shapes(loaded):
    _, _, _, mean_entry, std_entry, mean_x, std_x, _ = loaded
    assert mean_entry.shape == std_entry.shape == (4,)
    assert mean_x.shape == std_x.shape == ()


def test_loader_is_reproducible(at_root, loaded):
    again = loader()
    for first, second in zip(loaded[:3], again[:3]):
        torch.testing.assert_close(tensors(first)[0], tensors(second)[0])
