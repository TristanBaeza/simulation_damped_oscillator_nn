import numpy as np
from CONSTANTS import (
    END_TIME,
    INIT_TIME,
    N_POINTS,
    N_SIMULATIONS,
    V0_MAX,
    V0_MIN,
    X0_MAX,
    X0_MIN,
)
from scipy.integrate import solve_ivp


class Simulation:
    def __init__(self):
        self.t_eval = np.linspace(INIT_TIME, END_TIME, N_POINTS)

    def oscillator(
        self, etat: tuple[float, float], omega0: float, ksi: float
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

    def creation_box(self) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        x0_list = np.linspace(X0_MIN, X0_MAX, N_SIMULATIONS)
        np.random.shuffle(x0_list)
        v0_list = np.linspace(V0_MIN, V0_MAX, N_SIMULATIONS)
        np.random.shuffle(v0_list)
