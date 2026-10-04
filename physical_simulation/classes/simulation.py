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


class Simulation:
    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.t_eval = np.linspace(INIT_TIME, END_TIME, N_POINTS)
        x0_list, v0_list, ksi_list, omega0_list = self.creation_box()
        self.x0_list = x0_list
        self.v0_list = v0_list
        self.ksi_list = ksi_list
        self.omega0_list = omega0_list

    def oscillator(
        self, t: float, etat: tuple[float, float], omega0: float, ksi: float
    ) -> list[float]:
        x, v = etat
        return [v, -2 * ksi * omega0 * v - omega0**2 * x]

    def solving_oscillator(self, omega0: float, ksi: float, x0, v0) -> np.ndarray:
        solution = solve_ivp(
            self.oscillator,
            (INIT_TIME, END_TIME),
            [x0, v0],
            args=[omega0, ksi],
            t_eval=self.t_eval,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )
        x, _ = solution.y
        return x

    def creation_box(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x0_list = np.linspace(X0_MIN, X0_MAX, N_SIMULATIONS)
        self.rng.shuffle(x0_list)
        v0_list = np.linspace(V0_MIN, V0_MAX, N_SIMULATIONS)
        self.rng.shuffle(v0_list)
        ksi_list = np.geomspace(KSI_MIN, KSI_MAX, N_SIMULATIONS)
        self.rng.shuffle(ksi_list)
        omega0_list = np.geomspace(OMEGA0_MIN, OMEGA0_MAX, N_SIMULATIONS)
        self.rng.shuffle(omega0_list)
        return x0_list, v0_list, ksi_list, omega0_list

    def run_all(self) -> np.ndarray:
        x = np.empty((N_SIMULATIONS, N_POINTS))
        for i in range(N_SIMULATIONS):
            x[i] = self.solving_oscillator(
                self.omega0_list[i], self.ksi_list[i], self.x0_list[i], self.v0_list[i]
            )
        return x

    def save(self, path: str = DATA_PATH) -> np.ndarray:
        """Lance toutes les simulations et les sauvegarde dans un fichier .npz."""
        x = self.run_all()
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            file_path,
            x=x,
            t=self.t_eval,
            x0=self.x0_list,
            v0=self.v0_list,
            ksi=self.ksi_list,
            omega0=self.omega0_list,
            seed=self.seed,
        )
        return x


if __name__ == "__main__":
    start = time.perf_counter()
    Simulation().save()
    elapsed = time.perf_counter() - start
    print(f"Simulations sauvegardées dans {DATA_PATH} en {elapsed:.1f} s")
