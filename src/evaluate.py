import torch
from data import get_dataloaders
from model import get_model
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
_, val_loader = get_dataloaders('data/train', 'data/val')

# Load model
model = get_model()
model.load_state_dict(torch.load("models/best_model.pth"))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Report:")
print(classification_report(all_labels, all_preds, target_names=["benign", "malignant"]))


with open("logs/eval.txt", "w") as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(str(cm))