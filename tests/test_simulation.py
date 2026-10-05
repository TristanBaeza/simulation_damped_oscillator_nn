import numpy as np
import pytest

import physical_simulation.classes.simulation as simulation_module
from CONSTANTS import KSI_MAX, KSI_MIN, N_POINTS, N_SIMULATIONS, OMEGA0_MAX, OMEGA0_MIN
from physical_simulation.classes.simulation import Simulation


def analytic_underdamped(t, omega0, ksi, x0, v0):
    omega_d = omega0 * np.sqrt(1 - ksi**2)
    envelope = np.exp(-ksi * omega0 * t)
    return envelope * (
        x0 * np.cos(omega_d * t)
        + (v0 + ksi * omega0 * x0) / omega_d * np.sin(omega_d * t)
    )


def test_oscillator_derivative():
    sim = Simulation()
    dx, dv = sim.oscillator(0.0, (1.0, 2.0), omega0=3.0, ksi=0.5)
    assert dx == 2.0
    assert dv == pytest.approx(-2 * 0.5 * 3.0 * 2.0 - 3.0**2 * 1.0)


@pytest.mark.parametrize(
    "omega0, ksi, x0, v0", [(2.0, 0.1, 1.0, 0.0), (0.7, 0.5, -0.3, 1.5)]
)
def test_solver_matches_analytic_solution(omega0, ksi, x0, v0):
    sim = Simulation()
    x = sim.solving_oscillator(omega0, ksi, x0, v0)
    expected = analytic_underdamped(sim.t_eval, omega0, ksi, x0, v0)
    assert x.shape == (N_POINTS,)
    np.testing.assert_allclose(x, expected, atol=1e-6)


def test_creation_box_ranges():
    sim = Simulation()
    assert len(sim.ksi_list) == N_SIMULATIONS
    assert sim.ksi_list.min() == pytest.approx(KSI_MIN)
    assert sim.ksi_list.max() == pytest.approx(KSI_MAX)
    assert sim.omega0_list.min() == pytest.approx(OMEGA0_MIN)
    assert sim.omega0_list.max() == pytest.approx(OMEGA0_MAX)


def test_same_seed_gives_same_parameters():
    a, b = Simulation(seed=1), Simulation(seed=1)
    np.testing.assert_array_equal(a.ksi_list, b.ksi_list)
    np.testing.assert_array_equal(a.x0_list, b.x0_list)


def test_different_seed_gives_different_parameters():
    a, b = Simulation(seed=1), Simulation(seed=2)
    assert not np.array_equal(a.ksi_list, b.ksi_list)


def test_parameters_are_not_paired_min_with_min():
    sim = Simulation()
    assert not np.array_equal(np.argsort(sim.ksi_list), np.argsort(sim.omega0_list))


def test_save_writes_consistent_file(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation_module, "N_SIMULATIONS", 5)
    sim = Simulation()
    path = tmp_path / "sub" / "sims.npz"
    x = sim.save(str(path))

    data = np.load(path)
    assert set(data.files) == {"x", "t", "x0", "v0", "ksi", "omega0", "seed"}
    assert data["x"].shape == (5, N_POINTS)
    np.testing.assert_array_equal(data["x"], x)
    np.testing.assert_array_equal(data["x"][:, 0], data["x0"])
    assert int(data["seed"]) == sim.seed
