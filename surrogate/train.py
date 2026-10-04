from torch import nn


def train_loop(dataloader, model, optimizer, device, loss_fn=nn.MSELoss()):
    size = len(dataloader.dataset)
    model.train()
    batch_size = dataloader.batch_size
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            current_loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {current_loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / batch
