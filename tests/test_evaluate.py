import torch
from unittest.mock import patch, MagicMock

from src.evaluate import collect_predictions, compute_metrics, evaluate


# -------------------------
# Dummy components
# -------------------------

class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Always predict class 0
        return torch.tensor([[10.0, 0.0]] * x.shape[0])


def get_dummy_loader():
    x = torch.randn(10, 3, 224, 224)
    y = torch.zeros(10, dtype=torch.long)  # all class 0
    dataset = list(zip(x, y))
    return torch.utils.data.DataLoader(dataset, batch_size=2)


# -------------------------
# Tests
# -------------------------

def test_collect_predictions():
    model = DummyModel()
    loader = get_dummy_loader()

    preds, labels = collect_predictions(model, loader, "cpu")

    assert len(preds) == len(labels)
    assert all(p == 0 for p in preds)
    assert all(l == 0 for l in labels)


def test_compute_metrics_perfect():
    labels = [0, 0, 0, 0]
    preds = [0, 0, 0, 0]

    metrics = compute_metrics(labels, preds)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"].shape == (1, 1) or metrics["confusion_matrix"].shape == (2, 2)


def test_compute_metrics_mixed():
    labels = [0, 1, 0, 1]
    preds = [0, 0, 1, 1]

    metrics = compute_metrics(labels, preds)

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_evaluate_pipeline():
    model = DummyModel()
    loader = get_dummy_loader()

    metrics = evaluate(model, loader, "cpu")

    assert "precision" in metrics
    assert "recall" in metrics
    assert "confusion_matrix" in metrics


def test_collect_predictions_device_cpu():
    model = DummyModel()
    loader = get_dummy_loader()

    preds, labels = collect_predictions(model, loader, "cpu")

    assert isinstance(preds, list)
    assert isinstance(labels, list)


def test_main_runs(tmp_path):
    fake_loader = [ (torch.randn(2,3,224,224), torch.zeros(2, dtype=torch.long)) ]

    fake_metrics = {
        "precision": 1.0,
        "recall": 1.0,
        "confusion_matrix": [[1,0],[0,1]],
        "report": "ok"
    }

    with patch("src.evaluate.get_dataloaders") as mock_data, \
         patch("src.evaluate.load_model") as mock_model, \
         patch("src.evaluate.evaluate") as mock_eval:

        mock_data.return_value = (None, fake_loader)
        mock_model.return_value = MagicMock()
        mock_eval.return_value = fake_metrics

        # Redirect logs folder
        with patch("builtins.open") as mock_open:
            from src.evaluate import main
            main()

            mock_data.assert_called_once()
            mock_model.assert_called_once()
            mock_eval.assert_called_once()