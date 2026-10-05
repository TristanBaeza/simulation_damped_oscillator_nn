import argparse
from copy import deepcopy
from pathlib import Path

from CONSTANTS import BATCH_SIZE, HIDDEN, N_EPOCHS, PATIENCE_MAX
from surrogate import train
from surrogate.data import loader
from surrogate.model import MLP
import torch


def model_file_name():
    value = HIDDEN[0]
    multiplicator = 0
    name = "models/mlp_"
    for element in HIDDEN:  # (128, 128, 64) -> "128x2_64x1"
        if element == value:
            multiplicator += 1
        else:
            name = name + f"{value}x{multiplicator}_"
            multiplicator = 1
            value = element
    name = name + f"{value}x{multiplicator}_"
    return name + f"bs{BATCH_SIZE}.pt"


def main(n_epochs=N_EPOCHS, patience_max=PATIENCE_MAX):
    (
        train_loader,
        val_loader,
        test_loader,
        mean_entry,
        std_entry,
        mean_x,
        std_x,
        device,
        t,
    ) = loader()
    X, y = next(iter(train_loader))  # one batch, to read the layer sizes
    dimension_in, dimension_out = X.shape[1], y.shape[1]
    model = MLP(dimension_in, dimension_out, HIDDEN)
    model = model.to(device)
    patience = 0
    optimizer = torch.optim.Adam(model.parameters())
    loss_list_train = []
    loss_list_val = []
    loss_val = float(torch.inf)
    for epoch in range(n_epochs):
        training_loss = train.train_loop(train_loader, model, optimizer, device)
        loss_list_train.append(training_loss)
        temp = train.test_loop(val_loader, model, device)
        loss_list_val.append(temp)
        if temp < loss_val:
            patience = 0
            best_model = deepcopy(model.state_dict())  # state_dict holds references
            loss_val = temp
        else:
            patience += 1
        print(
            f"epoch {epoch + 1:>4d} | train {training_loss:.6f} | val {temp:.6f} "
            f"| patience {patience}/{patience_max}"
        )
        if patience >= patience_max:
            break
    total_model = {
        "best_model": best_model,
        "mean_entry": mean_entry,
        "std_entry": std_entry,
        "mean_x": mean_x,
        "std_x": std_x,
        "HIDDEN": HIDDEN,
        "dimension_in": dimension_in,
        "dimension_out": dimension_out,
        "loss_list_train": loss_list_train,
        "loss_list_val": loss_list_val,
        "t": t,
    }
    file_path = Path(model_file_name())
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(total_model, file_path)
    model.load_state_dict(best_model)

    print(f"best val loss: {loss_val:.6f}")
    print(f"test loss: {train.test_loop(test_loader, model, device):.6f}")
    print(f"model saved in {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the surrogate network.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="max epochs")
    parser.add_argument(
        "--patience", type=int, default=PATIENCE_MAX, help="early stopping patience"
    )
    args = parser.parse_args()
    main(args.epochs, args.patience)
