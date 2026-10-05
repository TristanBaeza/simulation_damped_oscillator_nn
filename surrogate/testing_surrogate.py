import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from CONSTANTS import MODEL_PATH
from physical_simulation.simulation import (
    run_simulations,
    sample_parameters,
    solve_oscillator,
)
from surrogate.model import MLP

SOLVER_COLOR = "#2a78d6"
NETWORK_COLOR = "#eb6834"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE_COLOR = "#fcfcfb"


def load_surrogate(model_path=MODEL_PATH):
    total_model = torch.load(model_path, map_location="cpu")
    model = MLP(
        total_model["dimension_in"], total_model["dimension_out"], total_model["HIDDEN"]
    )
    model.load_state_dict(total_model["best_model"])
    model.eval()
    return model, total_model


def predict(model, total_model, parameters):
    X = torch.tensor(parameters, dtype=torch.float32)
    X[:, 2] = torch.log10(X[:, 2])  # same preprocessing as in data.py
    X[:, 3] = torch.log10(X[:, 3])
    X = (X - total_model["mean_entry"]) / total_model["std_entry"]
    with torch.no_grad():
        pred = model(X)
    return (pred * total_model["std_x"] + total_model["mean_x"]).numpy()  # denormalize


def best_time(function, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)  # least disturbed run


def style_axes(ax):
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def testing(x0, v0, ksi, omega0, model_path=MODEL_PATH, n_batch=100):
    model, total_model = load_surrogate(model_path)
    t = total_model["t"].double().numpy()  # grid the model was trained on
    parameters = np.array([[x0, v0, ksi, omega0]])

    predicted_x = predict(model, total_model, parameters)[0]
    simulated_x = solve_oscillator(x0, v0, ksi, omega0, t)

    error = predicted_x - simulated_x
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    amplitude = np.max(np.abs(simulated_x))
    print(f"Paramètres : x0={x0}, v0={v0}, ksi={ksi}, omega0={omega0}")
    print(f"Erreur RMS : {rmse:.4f} ({rmse / amplitude:.1%} de l'amplitude)")
    print(f"Erreur max : {max_error:.4f} ({max_error / amplitude:.1%} de l'amplitude)")

    predict(model, total_model, parameters)  # warm-up: the first call is slower
    x0_batch, v0_batch, ksi_batch, omega0_batch = sample_parameters(
        n_simulations=n_batch, seed=0
    )
    batch = np.stack([x0_batch, v0_batch, ksi_batch, omega0_batch], axis=1)
    time_solver_one = best_time(
        lambda: solve_oscillator(x0, v0, ksi, omega0, t), repeats=5
    )
    time_network_one = best_time(
        lambda: predict(model, total_model, parameters), repeats=20
    )
    time_solver_batch = best_time(
        lambda: run_simulations(x0_batch, v0_batch, ksi_batch, omega0_batch, t),
        repeats=1,
    )
    time_network_batch = best_time(
        lambda: predict(model, total_model, batch), repeats=5
    )

    print()
    print(f"{'':<22}{'solveur':>12}{'réseau':>12}{'accélération':>15}")
    print(
        f"{'1 simulation':<22}{time_solver_one * 1e3:>9.2f} ms"
        f"{time_network_one * 1e3:>9.2f} ms{time_solver_one / time_network_one:>14.0f}x"
    )
    print(
        f"{f'{n_batch} simulations':<22}{time_solver_batch * 1e3:>9.1f} ms"
        f"{time_network_batch * 1e3:>9.2f} ms"
        f"{time_solver_batch / time_network_batch:>14.0f}x"
    )

    fig = plt.figure(figsize=(11, 7), facecolor=SURFACE_COLOR)
    grid = fig.add_gridspec(2, 2, height_ratios=(2, 1))
    ax_curve = fig.add_subplot(grid[0, :])
    ax_error = fig.add_subplot(grid[1, 0], sharex=ax_curve)
    ax_loss = fig.add_subplot(grid[1, 1])

    ax_curve.plot(t, simulated_x, color=SOLVER_COLOR, linewidth=2, label="Solveur")
    ax_curve.plot(
        t, predicted_x, color=NETWORK_COLOR, linewidth=2, linestyle="--", label="Réseau"
    )
    ax_curve.set_title(
        f"x(t) pour x0={x0}, v0={v0}, ξ={ksi}, ω0={omega0}",
        loc="left",
        color=TEXT_COLOR,
    )
    ax_curve.set_ylabel("x")
    ax_curve.legend(frameon=False, labelcolor=TEXT_COLOR)

    ax_error.plot(t, error, color=NETWORK_COLOR, linewidth=1.5)
    ax_error.axhline(0, color=TEXT_COLOR, linewidth=0.8)
    ax_error.set_title(
        f"Erreur réseau − solveur (RMS {rmse:.3f})", loc="left", color=TEXT_COLOR
    )
    ax_error.set_xlabel("t (s)")
    ax_error.set_ylabel("erreur")

    epochs = np.arange(1, len(total_model["loss_list_train"]) + 1)
    ax_loss.plot(
        epochs, total_model["loss_list_train"], color=SOLVER_COLOR, linewidth=2,
        label="Entraînement",
    )
    ax_loss.plot(
        epochs, total_model["loss_list_val"], color=NETWORK_COLOR, linewidth=2,
        linestyle="--", label="Validation",
    )
    ax_loss.set_yscale("log")
    ax_loss.set_title("Courbe d'apprentissage (MSE normalisée)", loc="left", color=TEXT_COLOR)
    ax_loss.set_xlabel("époque")
    ax_loss.legend(frameon=False, labelcolor=TEXT_COLOR)

    for ax in (ax_curve, ax_error, ax_loss):
        style_axes(ax)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the network with the solver for one parameter set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--x0", type=float, default=1.0, help="initial position")
    parser.add_argument("--v0", type=float, default=0.0, help="initial velocity")
    parser.add_argument("--ksi", type=float, default=0.1, help="damping ratio")
    parser.add_argument(
        "--omega0", type=float, default=2.0, help="natural frequency (rad/s)"
    )
    parser.add_argument("--model", default=MODEL_PATH, help="checkpoint to load")
    parser.add_argument(
        "--n-batch", type=int, default=100, help="batch size for the timing"
    )
    args = parser.parse_args()
    testing(args.x0, args.v0, args.ksi, args.omega0, args.model, args.n_batch)
