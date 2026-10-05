from torch import nn
import torch


def train_loop(dataloader, model, optimizer, device, loss_fn=nn.MSELoss()):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * len(X)  # weighted: the last batch can be smaller

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / size


def test_loop(dataloader, model, device, loss_fn=nn.MSELoss()):
    model.eval()
    size = len(dataloader.dataset)
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * len(X)
    return total_loss / size
