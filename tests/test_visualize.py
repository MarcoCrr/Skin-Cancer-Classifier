import os
import numpy as np
import torch
import pytest
from pathlib import Path

from src.visualize import load_config, load_training_log
from src.visualize import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_training_curves,
    plot_precision_recall_curve,
)
from src.visualize import get_predictions, plot_predictions
from src.visualize import validate_checkpoint, load_model


def test_load_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("system:\n  device: cpu\n")

    config = load_config(config_file)

    assert config["system"]["device"] == "cpu"


def test_load_training_log(tmp_path):
    log_file = tmp_path / "log.txt"
    log_file.write_text("0,0.5,0.6\n1,0.4,0.7\n")

    epochs, losses, accs = load_training_log(log_file)

    assert epochs == [0, 1]
    assert losses == [0.5, 0.4]
    assert accs == [0.6, 0.7]


def test_plot_roc_curve(tmp_path):
    labels = [0, 1, 0, 1]
    probs = [0.1, 0.9, 0.2, 0.8]

    out = tmp_path / "roc.png"
    auc_score = plot_roc_curve(labels, probs, save_path=out)

    assert 0 <= auc_score <= 1
    assert out.exists()


def test_plot_precision_recall_curve(tmp_path):
    labels = [0, 1, 0, 1]
    probs = [0.1, 0.9, 0.2, 0.8]

    out = tmp_path / "pr.png"
    ap = plot_precision_recall_curve(labels, probs, save_path=out)

    assert 0 <= ap <= 1
    assert out.exists()


def test_plot_confusion_matrix(tmp_path):
    cm = np.array([[2, 1], [0, 3]])
    out = tmp_path / "cm.png"

    plot_confusion_matrix(cm, ["a", "b"], save_path=out)

    assert out.exists()


def test_plot_training_curves(tmp_path):
    epochs = [0, 1, 2]
    losses = [0.5, 0.4, 0.3]
    accs = [0.6, 0.7, 0.8]

    out = tmp_path / "curves.png"

    plot_training_curves(epochs, losses, accs, save_path=out)

    assert out.exists()

#################################################################################

class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0.1, 0.9]] * x.shape[0])
    
def dummy_loader():
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    return [(images, labels)]

def test_get_predictions():
    model = DummyModel()
    loader = dummy_loader()

    images, preds, labels, confs, probs = get_predictions(model, loader, "cpu")

    assert len(images) == 4
    assert len(preds) == 4
    assert len(labels) == 4
    assert len(confs) == 4
    assert len(probs) == 4


def test_plot_predictions(tmp_path):
    images = [torch.randn(3, 224, 224) for _ in range(4)]
    preds = [0, 1, 0, 1]
    labels = [0, 0, 0, 1]
    confs = [0.9, 0.8, 0.7, 0.6]

    out = tmp_path / "preds.png"

    plot_predictions(
        images, preds, labels, confs,
        class_names=["a", "b"],
        save_path=out
    )

    assert out.exists()


def test_validate_checkpoint_valid():
    checkpoint = {
        "model_state_dict": {},
        "num_classes": 2
    }

    validate_checkpoint(checkpoint)