import argparse
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def map_label(dx):
    """Map diagnosis to binary class."""
    return "malignant" if dx == "mel" else "benign"


def create_dirs(base_path):
    for split in ["train", "val"]:
        for cls in ["benign", "malignant"]:
            (base_path / split / cls).mkdir(parents=True, exist_ok=True)


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    metadata_path = data_dir / "HAM10000_metadata.csv"
    img_dirs = [
        data_dir / "HAM10000_images_part_1",
        data_dir / "HAM10000_images_part_2"
    ]

    print("Loading metadata...")
    df = pd.read_csv(metadata_path)

    # Map labels
    df["label"] = df["dx"].apply(map_label)

    # Optional subsampling
    if args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)

    # Train/val split
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_split,
        stratify=df["label"],
        random_state=42
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # Create output folders
    create_dirs(output_dir)

    def find_image(image_id):
        for d in img_dirs:
            path = d / f"{image_id}.jpg"
            if path.exists():
                return path
        return None

    def copy_images(df_subset, split):
        for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"{split}"):
            img_path = find_image(row["image_id"])
            if img_path is None:
                continue

            target_dir = output_dir / split / row["label"]
            target_path = target_dir / img_path.name

            shutil.copy(img_path, target_path)

    print("Copying training images...")
    copy_images(train_df, "train")

    print("Copying validation images...")
    copy_images(val_df, "val")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAM10000 dataset")

    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to raw dataset")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for processed data")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--sample_size", type=int, default=2000,
                        help="Number of samples to use (None = all)")
    args = parser.parse_args()
    main(args)