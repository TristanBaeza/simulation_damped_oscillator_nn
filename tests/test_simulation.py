import numpy as np
import pytest

from CONSTANTS import KSI_MAX, KSI_MIN, N_POINTS, N_SIMULATIONS, OMEGA0_MAX, OMEGA0_MIN
from physical_simulation.simulation import (
    generate_dataset,
    load_dataset,
    oscillator,
    run_simulations,
    sample_parameters,
    solve_oscillator,
    time_grid,
)


def analytic_underdamped(t, x0, v0, ksi, omega0):
    omega_d = omega0 * np.sqrt(1 - ksi**2)
    envelope = np.exp(-ksi * omega0 * t)
    return envelope * (
        x0 * np.cos(omega_d * t)
        + (v0 + ksi * omega0 * x0) / omega_d * np.sin(omega_d * t)
    )


def test_oscillator_derivative():
    dx, dv = oscillator(0.0, (1.0, 2.0), omega0=3.0, ksi=0.5)
    assert dx == 2.0
    assert dv == pytest.approx(-2 * 0.5 * 3.0 * 2.0 - 3.0**2 * 1.0)


def test_time_grid():
    t = time_grid(0.0, 10.0, 11)
    np.testing.assert_allclose(t, np.arange(11.0))
    assert time_grid().shape == (N_POINTS,)


@pytest.mark.parametrize(
    "x0, v0, ksi, omega0", [(1.0, 0.0, 0.1, 2.0), (-0.3, 1.5, 0.5, 0.7)]
)
def test_solver_matches_analytic_solution(x0, v0, ksi, omega0):
    t = time_grid()
    x = solve_oscillator(x0, v0, ksi, omega0, t)
    assert x.shape == (N_POINTS,)
    np.testing.assert_allclose(x, analytic_underdamped(t, x0, v0, ksi, omega0), atol=1e-6)


def test_sample_parameters_ranges():
    x0_list, v0_list, ksi_list, omega0_list = sample_parameters()
    assert len(ksi_list) == N_SIMULATIONS
    assert ksi_list.min() == pytest.approx(KSI_MIN)
    assert ksi_list.max() == pytest.approx(KSI_MAX)
    assert omega0_list.min() == pytest.approx(OMEGA0_MIN)
    assert omega0_list.max() == pytest.approx(OMEGA0_MAX)


def test_sample_parameters_custom_settings():
    _, _, ksi_list, _ = sample_parameters(n_simulations=7, ksi_range=(0.1, 1.0))
    assert len(ksi_list) == 7
    assert ksi_list.min() == pytest.approx(0.1)
    assert ksi_list.max() == pytest.approx(1.0)


def test_same_seed_gives_same_parameters():
    for a, b in zip(sample_parameters(seed=1), sample_parameters(seed=1)):
        np.testing.assert_array_equal(a, b)


def test_different_seed_gives_different_parameters():
    assert not np.array_equal(sample_parameters(seed=1)[2], sample_parameters(seed=2)[2])


def test_parameters_are_not_paired_min_with_min():
    _, _, ksi_list, omega0_list = sample_parameters()
    assert not np.array_equal(np.argsort(ksi_list), np.argsort(omega0_list))


def test_run_simulations_matches_single_solves():
    t = time_grid()
    params = sample_parameters(n_simulations=3)
    x = run_simulations(*params, t)
    assert x.shape == (3, N_POINTS)
    for i in range(3):
        np.testing.assert_array_equal(
            x[i], solve_oscillator(*(p[i] for p in params), t)
        )


def test_generate_and_load_dataset(tmp_path):
    path = tmp_path / "sub" / "sims.npz"
    x = generate_dataset(str(path), n_simulations=5, seed=3)

    data = load_dataset(str(path))
    assert set(data) == {"x", "t", "x0", "v0", "ksi", "omega0", "seed"}
    assert data["x"].shape == (5, N_POINTS)
    np.testing.assert_array_equal(data["x"], x)
    np.testing.assert_array_equal(data["x"][:, 0], data["x0"])
    assert int(data["seed"]) == 3
