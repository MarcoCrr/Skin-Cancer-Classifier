import torch
import matplotlib.pyplot as plt
from src.data import get_dataloaders
from src.model import get_model
import numpy as np
import argparse
import yaml
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score



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
    Supports both:
    - new format: dict with metadata
    - old format: raw state_dict

    I'm inserting this if/else since there might be old checkpoints
    without metadata. Backward compatibility.

    Args:
        model_path (str): Path to saved model.
        device (str): Device.

    Returns:
        torch.nn.Module: Loaded model.
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Case 1: new format (dict with metadata)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        num_classes = checkpoint.get("num_classes", 2)
        model = get_model(num_classes=num_classes)

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            raise RuntimeError(
                "Checkpoint is incompatible with current model."
            )

    # Case 2: old format (just state_dict)
    else:
        model = get_model()
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            raise RuntimeError(
                "Old checkpoint incompatible with current model. "
                "You likely changed architecture (e.g., dummy model vs ResNet)."
            )

    return model.to(device)


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
        tuple: (images, preds, labels, confidences, positive_class_probs)
    """
    model.eval()

    all_images, all_preds, all_labels, all_confs, all_probs = [], [], [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            positive_probs = probs[:, 1]

            all_images.extend(images.cpu())
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())
            all_confs.extend(confs.cpu())
            all_probs.extend(positive_probs.cpu().numpy())

    return all_images, all_preds, all_labels, all_confs, all_probs


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
    print(f"Predictions plot saved to {save_path}")


def plot_roc_curve(labels, probs, save_path="logs/roc_curve.png"):
    """
    Plot ROC curve.

    Args:
        labels (list or array): Ground truth labels (0/1).
        probs (list or array): Probabilities for positive class (class 1).
        save_path (str): Path to save the plot.

    Returns:
        float: Computed AUC score.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path} with AUC = {roc_auc:.2f}")

    return roc_auc


def plot_confusion_matrix(cm, class_names,
                          save_path="logs/confusion_matrix.png"):
    """
    Plot confusion matrix as heatmap.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): Class labels.
        save_path (str): Path to save plot.
    """
    plt.figure()

    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def load_training_log(log_path):
    """
    Load training log file.

    Expected format per line:
        epoch,train_loss,val_acc

    Args:
        log_path (str): Path to log file.

    Returns:
        tuple: (epochs, train_losses, val_accuracies)
    """
    epochs = []
    train_losses = []
    val_accuracies = []

    with open(log_path, "r") as f:
        for line in f:
            epoch, loss, acc = line.strip().split(",")
            epochs.append(int(epoch))
            train_losses.append(float(loss))
            val_accuracies.append(float(acc))

    return epochs, train_losses, val_accuracies


def plot_training_curves(epochs, train_losses, val_accuracies,
                         save_path="logs/training_curves.png"):
    """
    Plot training loss and validation accuracy over epochs.

    Args:
        epochs (list)
        train_losses (list)
        val_accuracies (list)
        save_path (str)
    """
    plt.figure()

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_precision_recall_curve(labels, probs,
                                save_path="logs/precision_recall_curve.png"):
    """
    Plot Precision-Recall curve.

    Args:
        labels (list or array): Ground truth labels (0/1).
        probs (list or array): Probabilities for positive class (class 1).
        save_path (str): Path to save the plot.

    Returns:
        float: Average precision (AP) score.
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap_score = average_precision_score(labels, probs)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap_score:.2f}")
    baseline = sum(labels) / len(labels)
    plt.hlines(baseline, 0, 1, linestyles="dashed", label="Baseline")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.savefig(save_path)
    plt.close()

    return ap_score


#################################################################

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

    # device = config["system"]["device"] if torch.cuda.is_available() else "cpu"
    device = 'cpu'

    _, val_loader = get_dataloaders(
        config["data"]["train_dir"],
        config["data"]["val_dir"],
        batch_size=8
    )

    model = load_model(model_path, device)

    images, preds, labels, confs, probs = get_predictions(model, val_loader, device)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    plot_confusion_matrix(cm, ["benign", "malignant"])
    plot_roc_curve(labels, probs)
    # plot_precision_recall_curve(labels, probs)

    plot_predictions(
        images, preds, labels, confs,
        class_names=["benign", "malignant"],
        mistakes_only=mistakes_only,
        max_images=num_images
    )
    try:
        epochs, losses, accs = load_training_log("logs/train_log.txt")
        plot_training_curves(epochs, losses, accs)
    except FileNotFoundError:
        print("Training log not found, skipping training curves.")


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