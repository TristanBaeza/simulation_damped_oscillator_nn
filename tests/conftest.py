from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "simulations.npz"


@pytest.fixture
def at_root(monkeypatch):
    monkeypatch.chdir(ROOT)  # data and model paths are relative to the root


requires_data = pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason="data/simulations.npz absent : lancer python -m physical_simulation.simulation",
)
