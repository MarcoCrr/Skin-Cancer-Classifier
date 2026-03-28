import torch
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

from src.data import get_dataloaders
from src.model import get_model


def collect_predictions(model, dataloader, device):
    """
    Run inference on a dataloader and collect predictions and labels.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device ("cpu" or "cuda").

    Returns:
        tuple: (all_preds, all_labels) as Python lists.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def compute_metrics(all_labels, all_preds):
    """
    Compute classification metrics.

    Args:
        all_labels (list): Ground truth labels.
        all_preds (list): Predicted labels.

    Returns:
        dict: Dictionary with precision, recall, confusion matrix, report.
    """

    labels = [0, 1]
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=["benign", "malignant"],
        zero_division=0
    )

    return {
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "report": report,
    }


def evaluate(model, dataloader, device):
    """
    Full evaluation pipeline: inference + metrics.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device.

    Returns:
        dict: Evaluation metrics.
    """
    preds, labels = collect_predictions(model, dataloader, device)
    return compute_metrics(labels, preds)


def load_model(model_path, device):
    """
    Load model weights from disk.

    Args:
        model_path (str): Path to saved model.
        device (str): Device.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


def main():
    """
    CLI entry point for evaluation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_dataloaders("data/train", "data/val")

    model = load_model("models/best_model.pth", device)

    metrics = evaluate(model, val_loader, device)

    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nDetailed Report:")
    print(metrics["report"])

    with open("logs/eval.txt", "w") as f:
        f.write(f"Precision: {metrics['precision']}\n")
        f.write(f"Recall: {metrics['recall']}\n")
        f.write(str(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()