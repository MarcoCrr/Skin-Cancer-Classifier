import torch
import pytest

from src.model import get_model

from unittest.mock import patch
import torchvision.models as models

# In order not to have to download pretrained weights during testing:
@patch("src.model.models.resnet18")
def test_model_creation_mocked(mock_resnet):
    mock_resnet.return_value = models.resnet18(weights=None)

    model = get_model()

    assert model is not None

# -------------------------
# Tests
# -------------------------


def test_model_output_shape():
    model = get_model(num_classes=3, freeze_backbone=True)

    x = torch.randn(4, 3, 224, 224)
    y = model(x)

    assert y.shape == (4, 3)


def test_final_layer_matches_num_classes():
    num_classes = 5
    model = get_model(num_classes=num_classes)

    assert model.fc.out_features == num_classes


def test_backbone_frozen():
    model = get_model(freeze_backbone=True)

    # All params except fc should be frozen
    frozen = []
    trainable = []

    for name, param in model.named_parameters():
        if "fc" in name:
            trainable.append(param.requires_grad)
        else:
            frozen.append(param.requires_grad)

    assert all(not f for f in frozen)
    assert all(t for t in trainable)


def test_backbone_not_frozen():
    model = get_model(freeze_backbone=False)

    params = [p.requires_grad for p in model.parameters()]

    assert all(params)  # everything trainable


def test_forward_pass_runs():
    model = get_model()

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    assert isinstance(y, torch.Tensor)


def test_model_has_resnet_structure():
    model = get_model()

    # Basic sanity checks
    assert hasattr(model, "fc")
    assert hasattr(model, "layer1")
    assert hasattr(model, "conv1")