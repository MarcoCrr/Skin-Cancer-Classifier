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


def test_plot_training_curves(tmp_path):
    epochs = [0, 1, 2]
    losses = [0.5, 0.4, 0.3]
    accs = [0.6, 0.7, 0.8]

    out = tmp_path / "curves.png"

    plot_training_curves(epochs, losses, accs, save_path=out)

    assert out.exists()


