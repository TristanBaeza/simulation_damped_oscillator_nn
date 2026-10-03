import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from CONSTANTS import SEED


def loader() -> tuple[
    DataLoader[tuple[torch.Tensor, ...]],
    DataLoader[tuple[torch.Tensor, ...]],
    DataLoader[tuple[torch.Tensor, ...]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.device,
]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data = np.load("data/simulations.npz")
    x = torch.tensor(data["x"], dtype=torch.float32)
    entry = torch.tensor(
        np.stack([data["x0"], data["v0"], data["ksi"], data["omega0"]], axis=1),
        dtype=torch.float32,
    )
    t = torch.tensor(data["t"], dtype=torch.float32)

    gen = torch.Generator().manual_seed(SEED)
    n = len(x)
    perm = torch.randperm(n, generator=gen)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    index_train = perm[:n_train]
    index_val = perm[n_train : n_train + n_val]
    index_test = perm[n_train + n_val :]

    entry[:, 2] = torch.log10(entry[:, 2])
    entry[:, 3] = torch.log10(entry[:, 3])

    mean_entry = entry[index_train].mean(dim=0)
    std_entry = entry[index_train].std(dim=0)

    entry = (entry - mean_entry) / std_entry

    mean_x = x[index_train].mean()
    std_x = x[index_train].std()

    x = (x - mean_x) / std_x

    train_dataset = TensorDataset(entry[index_train], x[index_train])
    val_dataset = TensorDataset(entry[index_val], x[index_val])
    test_dataset = TensorDataset(entry[index_test], x[index_test])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=gen)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        mean_entry,
        std_entry,
        mean_x,
        std_x,
        device,
    )
