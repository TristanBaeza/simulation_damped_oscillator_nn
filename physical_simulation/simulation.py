import argparse
import time
from pathlib import Path

import numpy as np
from CONSTANTS import (
    DATA_PATH,
    END_TIME,
    INIT_TIME,
    KSI_MAX,
    KSI_MIN,
    N_POINTS,
    N_SIMULATIONS,
    OMEGA0_MAX,
    OMEGA0_MIN,
    SEED,
    V0_MAX,
    V0_MIN,
    X0_MAX,
    X0_MIN,
)
from scipy.integrate import solve_ivp


def oscillator(
    t: float, etat: tuple[float, float], omega0: float, ksi: float
) -> list[float]:
    x, v = etat
    return [v, -2 * ksi * omega0 * v - omega0**2 * x]  # (x', v')


def time_grid(
    init_time: float = INIT_TIME, end_time: float = END_TIME, n_points: int = N_POINTS
) -> np.ndarray:
    return np.linspace(init_time, end_time, n_points)


def solve_oscillator(
    x0: float, v0: float, ksi: float, omega0: float, t_eval: np.ndarray
) -> np.ndarray:
    solution = solve_ivp(
        oscillator,
        (t_eval[0], t_eval[-1]),
        [x0, v0],
        args=(omega0, ksi),
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )
    x, _ = solution.y
    return x


def sample_parameters(
    n_simulations: int = N_SIMULATIONS,
    seed: int = SEED,
    x0_range: tuple[float, float] = (X0_MIN, X0_MAX),
    v0_range: tuple[float, float] = (V0_MIN, V0_MAX),
    ksi_range: tuple[float, float] = (KSI_MIN, KSI_MAX),
    omega0_range: tuple[float, float] = (OMEGA0_MIN, OMEGA0_MAX),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0_list = np.linspace(*x0_range, n_simulations)
    rng.shuffle(x0_list)  # independent shuffles: Latin hypercube
    v0_list = np.linspace(*v0_range, n_simulations)
    rng.shuffle(v0_list)
    ksi_list = np.geomspace(*ksi_range, n_simulations)
    rng.shuffle(ksi_list)
    omega0_list = np.geomspace(*omega0_range, n_simulations)
    rng.shuffle(omega0_list)
    return x0_list, v0_list, ksi_list, omega0_list


def run_simulations(
    x0_list: np.ndarray,
    v0_list: np.ndarray,
    ksi_list: np.ndarray,
    omega0_list: np.ndarray,
    t_eval: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    n = len(x0_list)
    x = np.empty((n, len(t_eval)))
    step = max(1, n // 20)  # progress every 5%
    start = time.perf_counter()
    for i in range(n):
        x[i] = solve_oscillator(
            x0_list[i], v0_list[i], ksi_list[i], omega0_list[i], t_eval
        )
        if verbose and ((i + 1) % step == 0 or i + 1 == n):
            elapsed = time.perf_counter() - start
            remaining = elapsed / (i + 1) * (n - i - 1)
            print(
                f"{i + 1:>{len(str(n))}d}/{n} simulations ({(i + 1) / n:.0%}) "
                f"| {elapsed:.1f} s écoulées | ~{remaining:.1f} s restantes",
                flush=True,
            )
    return x


def generate_dataset(
    path: str = DATA_PATH, n_simulations: int = N_SIMULATIONS, seed: int = SEED
) -> np.ndarray:
    t_eval = time_grid()
    x0_list, v0_list, ksi_list, omega0_list = sample_parameters(n_simulations, seed)
    x = run_simulations(
        x0_list, v0_list, ksi_list, omega0_list, t_eval, verbose=True
    )
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file_path,
        x=x,
        t=t_eval,
        x0=x0_list,
        v0=v0_list,
        ksi=ksi_list,
        omega0=omega0_list,
        seed=seed,
    )
    return x


def load_dataset(path: str = DATA_PATH) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the simulation dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--path", default=DATA_PATH, help="output .npz file")
    parser.add_argument(
        "--n-simulations", type=int, default=N_SIMULATIONS, help="number of curves"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="parameter sampling seed")
    args = parser.parse_args()

    start = time.perf_counter()
    generate_dataset(args.path, args.n_simulations, args.seed)
    elapsed = time.perf_counter() - start
    print(f"Simulations sauvegardées dans {args.path} en {elapsed:.1f} s")
