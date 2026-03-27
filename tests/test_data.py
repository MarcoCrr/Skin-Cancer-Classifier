import torch
from PIL import Image

from src.data import get_dataloaders


# -------------------------
# Helpers
# -------------------------

def create_dummy_image(path):
    img = Image.new("RGB", (10, 10), color="red")
    img.save(path)


def create_dataset_structure(base_dir):
    for split in ["train", "val"]:
        for cls in ["benign", "malignant"]:
            dir_path = base_dir / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)

            # create 2 dummy images per class
            for i in range(2):
                create_dummy_image(dir_path / f"img_{i}.jpg")


# -------------------------
# Tests
# -------------------------

def test_get_dataloaders_returns_loaders(tmp_path):
    create_dataset_structure(tmp_path)

    train_loader, val_loader = get_dataloaders(
        tmp_path / "train",
        tmp_path / "val",
        batch_size=2
    )

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)


def test_dataloader_batch_size(tmp_path):
    create_dataset_structure(tmp_path)

    batch_size = 2
    train_loader, _ = get_dataloaders(
        tmp_path / "train",
        tmp_path / "val",
        batch_size=batch_size
    )

    batch = next(iter(train_loader))
    images, labels = batch

    assert images.shape[0] == batch_size


def test_dataloader_output_shape(tmp_path):
    create_dataset_structure(tmp_path)

    train_loader, _ = get_dataloaders(
        tmp_path / "train",
        tmp_path / "val",
        batch_size=2
    )

    images, labels = next(iter(train_loader))

    # After Resize(224,224) + ToTensor → shape [B, C, H, W]
    assert images.shape[1:] == (3, 224, 224)
    assert labels.dtype == torch.int64


def test_class_labels(tmp_path):
    create_dataset_structure(tmp_path)

    train_loader, _ = get_dataloaders(
        tmp_path / "train",
        tmp_path / "val",
        batch_size=2
    )

    _, labels = next(iter(train_loader))

    # Should be 0 or 1 (benign/malignant)
    assert set(labels.tolist()).issubset({0, 1})