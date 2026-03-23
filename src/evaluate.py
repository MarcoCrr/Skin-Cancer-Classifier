import torch
from data import get_dataloaders
from model import get_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, val_loader = get_dataloaders('data/train', 'data/val')
model = get_model()
model = model.to(device)
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Final Accuracy:", correct / total)