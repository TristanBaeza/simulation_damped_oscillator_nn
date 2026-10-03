from torch import nn


def train_loop(dataloader, model, optimizer, device, loss_fn=nn.MSELoss()):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X.to(device)
        y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
