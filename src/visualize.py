import torch
import matplotlib.pyplot as plt
import numpy as np
from data import get_dataloaders
from model import get_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
_, val_loader = get_dataloaders('data/train', 'data/val', batch_size=8)

# Load model
model = get_model()
model.load_state_dict(torch.load("models/best_model.pth"))
model = model.to(device)
model.eval()

class_names = ["benign", "malignant"]


def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis("off")


images_shown = 0
max_images = 8

plt.figure(figsize=(12, 6))

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)

        for i in range(images.size(0)):
            


            images_shown += 1
            plt.subplot(2, 4, images_shown)

            imshow(images[i].cpu())

            pred = class_names[preds[i]]
            true = class_names[labels[i]]
            conf = confs[i].item()

            correct = preds[i] == labels[i]
            color = "green" if correct else "red"

            plt.title(f"P: {pred} ({conf:.2f})\nT: {true}", color=color)

            if images_shown == max_images:
                plt.tight_layout()
                plt.show()
                plt.savefig("logs/predictions.png")
                exit()