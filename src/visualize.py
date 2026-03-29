import torch
import matplotlib.pyplot as plt
from src.data import get_dataloaders
from src.model import get_model
import argparse
import yaml


def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to config file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path, device):
    """
    Load model weights from disk.

    Args:
        model_path (str): Path to saved model.
        device (str): Device.

    Returns:
        torch.nn.Module: Loaded model.
    """
    checkpoint = torch.load(model_path, map_location=device)

    model = get_model(num_classes=checkpoint["num_classes"])
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        raise RuntimeError(
            "Checkpoint is incompatible with current model. "
            "You likely changed the architecture."
        )

    return model.to(device)

# def load_model(model_path, device):
#     """
#     Load trained model from checkpoint.

#     Args:
#         model_path (str): Path to saved model.
#         device (str): Device.

#     Returns:
#         torch.nn.Module: Loaded model.
#     """
#     checkpoint = torch.load(model_path, map_location=device)
#     model = get_model(num_classes=checkpoint.get("num_classes", 2))
#     model.load_state_dict(checkpoint["model_state_dict"])
#     return model.to(device)


def imshow(img):
    """
    Convert tensor image to numpy and plot it.

    Args:
        img (torch.Tensor): Image tensor (C, H, W).
    """
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis("off")


def get_predictions(model, dataloader, device):
    """
    Run inference and return predictions, labels, and confidences.

    Args:
        model (torch.nn.Module)
        dataloader (DataLoader)
        device (str)

    Returns:
        tuple: (images, preds, labels, confidences)
    """
    model.eval()

    all_images, all_preds, all_labels, all_confs = [], [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            all_images.extend(images.cpu())
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())
            all_confs.extend(confs.cpu())

    return all_images, all_preds, all_labels, all_confs


def plot_predictions(images, preds, labels, confs,
                     class_names,
                     mistakes_only=False,
                     max_images=8,
                     save_path="logs/predictions.png"):
    """
    Plot model predictions.

    Args:
        images (list)
        preds (list)
        labels (list)
        confs (list)
        class_names (list)
        mistakes_only (bool)
        max_images (int)
        save_path (str)
    """
    plt.figure(figsize=(12, 6))

    images_shown = 0

    for i in range(len(images)):
        is_mistake = preds[i] != labels[i]

        if mistakes_only and not is_mistake:
            continue

        images_shown += 1
        plt.subplot(2, 4, images_shown)

        imshow(images[i])

        pred = class_names[int(preds[i])]
        true = class_names[int(labels[i])]
        conf = float(confs[i])

        color = "green" if preds[i] == labels[i] else "red"

        plt.title(f"P: {pred} ({conf:.2f})\nT: {true}", color=color)

        if images_shown == max_images:
            break

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_visualization(config_path, model_path,
                     mistakes_only=False,
                     num_images=8):
    """
    Full visualization pipeline.

    Args:
        config_path (str)
        model_path (str)
        mistakes_only (bool)
        num_images (int)
    """
    config = load_config(config_path)

    device = config["system"]["device"] if torch.cuda.is_available() else "cpu"

    _, val_loader = get_dataloaders(
        config["data"]["train_dir"],
        config["data"]["val_dir"],
        batch_size=8
    )

    model = load_model(model_path, device)

    images, preds, labels, confs = get_predictions(model, val_loader, device)

    plot_predictions(
        images, preds, labels, confs,
        class_names=["benign", "malignant"],
        mistakes_only=mistakes_only,
        max_images=num_images
    )


def main():
    """
    CLI entry point.
    """

    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument("--mistakes_only", action="store_true", help="Show only incorrect predictions")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to display")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")

    args = parser.parse_args()

    run_visualization(
        args.config,
        args.model,
        mistakes_only=args.mistakes_only,
        num_images=args.num_images
    )


if __name__ == "__main__":
    main()