import pytest
import torch

import surrogate.main as main_module
from conftest import requires_data


@pytest.mark.parametrize(
    "hidden, expected",
    [
        ((128, 128, 128), "models/mlp_128x3_bs32.pt"),
        ((256, 128, 128, 64), "models/mlp_256x1_128x2_64x1_bs32.pt"),
        ((64,), "models/mlp_64x1_bs32.pt"),
        ((64, 32, 64), "models/mlp_64x1_32x1_64x1_bs32.pt"),
    ],
)
def test_model_file_name(monkeypatch, hidden, expected):
    monkeypatch.setattr(main_module, "HIDDEN", hidden)
    monkeypatch.setattr(main_module, "BATCH_SIZE", 32)
    assert main_module.model_file_name() == expected


@requires_data
def test_main_saves_loadable_checkpoint(at_root, tmp_path, monkeypatch):
    path = tmp_path / "models" / "test.pt"
    monkeypatch.setattr(main_module, "model_file_name", lambda: str(path))

    main_module.main(n_epochs=2)

    checkpoint = torch.load(path)
    assert len(checkpoint["loss_list_train"]) == 2
    assert len(checkpoint["loss_list_val"]) == 2
    model = main_module.MLP(
        checkpoint["dimension_in"], checkpoint["dimension_out"], checkpoint["HIDDEN"]
    )
    model.load_state_dict(checkpoint["best_model"])
    assert checkpoint["t"].shape == (checkpoint["dimension_out"],)
