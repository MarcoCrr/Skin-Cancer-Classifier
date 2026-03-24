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