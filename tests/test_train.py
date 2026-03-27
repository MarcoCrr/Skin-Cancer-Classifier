import json
import torch
import pytest
from unittest.mock import patch, MagicMock

from src.train import train


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def dummy_config():
    return {
        "system": {"device": "cpu"},
        "data": {
            "train_dir": "dummy_train",
            "val_dir": "dummy_val",
            "batch_size": 4
        },
        "training": {
            "epochs": 3,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "early_stopping_patience": 2
        }
    }


@pytest.fixture
def dummy_loader():
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = list(zip(x, y))
    return torch.utils.data.DataLoader(dataset, batch_size=5)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# -------------------------
# Tests
# -------------------------

@patch("src.train.get_dataloaders")
@patch("src.train.get_model")
@patch("src.train.torch.save")
def test_train_runs_and_returns_accuracy(
    mock_save, mock_get_model, mock_get_dataloaders, dummy_config, dummy_loader
):
    mock_get_dataloaders.return_value = (dummy_loader, dummy_loader)
    mock_get_model.return_value = DummyModel()

    with patch("src.train.evaluate", return_value=0.5):
        with patch("src.train.train_one_epoch", return_value=1.0):
            with patch("src.train.should_stop_early", return_value=(True, 0)):
                best_acc = train(dummy_config)

    assert best_acc == 0.5
    assert mock_save.called


@patch("src.train.get_dataloaders")
@patch("src.train.get_model")
@patch("src.train.torch.save")
def test_model_saved_only_on_improvement(
    mock_save, mock_get_model, mock_get_dataloaders, dummy_config, dummy_loader
):
    mock_get_dataloaders.return_value = (dummy_loader, dummy_loader)
    mock_get_model.return_value = DummyModel()

    # First call improves, second does not
    should_stop_side_effect = [(True, 0), (False, 1), (False, 2)]

    with patch("src.train.evaluate", return_value=0.5), \
         patch("src.train.train_one_epoch", return_value=1.0), \
         patch("src.train.should_stop_early", side_effect=should_stop_side_effect):

        train(dummy_config)

    # Should save only once (first improvement)
    assert mock_save.call_count == 1


@patch("src.train.get_dataloaders")
@patch("src.train.get_model")
def test_early_stopping_triggers(
    mock_get_model, mock_get_dataloaders, dummy_config, dummy_loader
):
    mock_get_dataloaders.return_value = (dummy_loader, dummy_loader)
    mock_get_model.return_value = DummyModel()

    dummy_config["training"]["epochs"] = 10
    dummy_config["training"]["early_stopping_patience"] = 1

    # Force no improvement → counter increases
    with patch("src.train.evaluate", return_value=0.5), \
         patch("src.train.train_one_epoch", return_value=1.0), \
         patch("src.train.should_stop_early", return_value=(False, 2)):

        best_acc = train(dummy_config)

    # Should stop early and never improve
    assert best_acc == 0


@patch("src.train.get_dataloaders")
@patch("src.train.get_model")
def test_device_fallback_to_cpu(
    mock_get_model, mock_get_dataloaders, dummy_config, dummy_loader
):
    mock_get_dataloaders.return_value = (dummy_loader, dummy_loader)
    mock_get_model.return_value = DummyModel()

    dummy_config["system"]["device"] = "cuda"

    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.train.evaluate", return_value=0.5), \
         patch("src.train.train_one_epoch", return_value=1.0), \
         patch("src.train.should_stop_early", return_value=(True, 0)):

        best_acc = train(dummy_config)

    assert best_acc == 0.5


@patch("src.train.get_dataloaders")
@patch("src.train.get_model")
def test_train_loop_multiple_epochs(
    mock_get_model, mock_get_dataloaders, dummy_config, dummy_loader
):
    mock_get_dataloaders.return_value = (dummy_loader, dummy_loader)
    mock_get_model.return_value = DummyModel()

    dummy_config["training"]["epochs"] = 3

    with patch("src.train.evaluate", return_value=0.6) as mock_eval, \
         patch("src.train.train_one_epoch", return_value=1.0) as mock_train_epoch, \
         patch("src.train.should_stop_early", return_value=(True, 0)):

        train(dummy_config)

    # Ensure loop actually ran multiple times
    assert mock_eval.call_count == 3
    assert mock_train_epoch.call_count == 3