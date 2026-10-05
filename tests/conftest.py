from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "simulations.npz"


@pytest.fixture
def at_root(monkeypatch):
    """Exécute le test depuis la racine du projet (les chemins y sont relatifs)."""
    monkeypatch.chdir(ROOT)


requires_data = pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason="data/simulations.npz absent : lancer python -m physical_simulation.classes.simulation",
)
