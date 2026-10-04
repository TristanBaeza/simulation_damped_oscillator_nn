from copy import deepcopy

from CONSTANTS import N_EPOCHS, PATIENCE_MAX
from surrogate import train
from surrogate.data import loader
from surrogate.model import MLP
import torch


def main():
    (
        train_loader,
        val_loader,
        test_loader,
        mean_entry,
        std_entry,
        mean_x,
        std_x,
        device,
    ) = loader()
    X, y = next(iter(train_loader))
    dimension_in, dimension_out = X.shape[1], y.shape[1]
    model = MLP(dimension_in, dimension_out)
    model = model.to(device)
    patience = 0
    optimizer = torch.optim.Adam(model.parameters())
    loss_list_train = []
    loss_list_val = []
    loss_val = float(torch.inf)
    for epoch in range(N_EPOCHS):
        if patience >= PATIENCE_MAX:
            break
        training_loss = train.train_loop(train_loader, model, optimizer, device)
        loss_list_train.append(training_loss)
        temp = train.test_loop(val_loader, model, device)
        loss_list_val.append(temp)
        if temp < loss_val:
            patience = 0
            best_model = deepcopy(model.state_dict())
        else:
            patience += 1
    best_model["mean_entry"] = mean_entry
    best_model["std_entry"] = std_entry
    best_model["mean_x"] = mean_x
    best_model["std_x"] = std_x
    torch.save(
        best_model,
    )
