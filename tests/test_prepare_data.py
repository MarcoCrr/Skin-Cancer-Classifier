import pytest
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

from src.prepare_data import map_label, create_dirs, main


# -------------------------
# Unit tests (pure functions)
# -------------------------

def test_map_label():
    assert map_label("mel") == "malignant"
    assert map_label("nv") == "benign"
    assert map_label("bcc") == "benign"


def test_create_dirs(tmp_path):
    create_dirs(tmp_path)

    for split in ["train", "val"]:
        for cls in ["benign", "malignant"]:
            assert (tmp_path / split / cls).exists()


# -------------------------
# Integration test for main
# -------------------------

def test_main_creates_split_dataset(tmp_path):
    # Create fake dataset structure
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"

    img_dir1 = data_dir / "HAM10000_images_part_1"
    img_dir2 = data_dir / "HAM10000_images_part_2"

    img_dir1.mkdir(parents=True)
    img_dir2.mkdir(parents=True)

    # Create fake images
    image_ids = ["img1", "img2", "img3", "img4"]
    for img_id in image_ids:
        (img_dir1 / f"{img_id}.jpg").write_bytes(b"fake")

    # Create metadata CSV
    df = pd.DataFrame({
        "image_id": image_ids,
        "dx": ["mel", "nv", "mel", "nv"]
    })

    metadata_path = data_dir / "HAM10000_metadata.csv"
    df.to_csv(metadata_path, index=False)

    # Create args
    args = SimpleNamespace(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        val_split=0.5,
        sample_size=None
    )

    # Run main
    main(args)

    # Check output structure
    for split in ["train", "val"]:
        for cls in ["benign", "malignant"]:
            assert (output_dir / split / cls).exists()

    # Check that some files were copied
    total_files = list(output_dir.rglob("*.jpg"))
    assert len(total_files) > 0


def test_main_handles_missing_images(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"

    img_dir = data_dir / "HAM10000_images_part_1"
    img_dir.mkdir(parents=True)

    # Metadata references non-existing image
    df = pd.DataFrame({
        "image_id": ["missing_img"],
        "dx": ["mel"]
    })

    metadata_path = data_dir / "HAM10000_metadata.csv"
    df.to_csv(metadata_path, index=False)

    args = SimpleNamespace(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        val_split=0.5,
        sample_size=None
    )

    # Should not crash
    main(args)

    # No images copied
    assert len(list(output_dir.rglob("*.jpg"))) == 0