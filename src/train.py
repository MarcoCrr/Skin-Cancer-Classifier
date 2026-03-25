import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import get_dataloaders
from model import get_model
import yaml
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config.yaml")
args = parser.parse_args()

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
with open("logs/config_used.json", "w") as f: #TODO: add date or whatever to distinguish the files
    json.dump(config, f, indent=4)


device = config["system"]["device"] if torch.cuda.is_available() else 'cpu'

train_loader, val_loader = get_dataloaders(config["data"]["train_dir"], config["data"]["val_dir"])

model = get_model()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(),
                       lr=config["training"]["learning_rate"],
                       weight_decay=config["training"]["weight_decay"])

epochs = config["training"]["epochs"]  # 3, toy project


best_acc = 0
train_losses = [] # TODO: instead of appending, do smt more memory-friendly
val_accuracies = []

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    
    train_losses.append(loss)
    
    # run validation for each epoch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct/total
    val_accuracies.append(val_acc)

    patience = config["training"]["early_stopping_patience"]
    counter = 0

    print(f"Validation Accuracy: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "models/best_model.pth")
        print("Saved best model")
    else:
        counter += 1

    with open("logs/train_log.txt", "a") as f: #TODO: add date or whatever to distinguish the files
        f.write(f"Epoch {epoch}, Loss: {loss.item()}, Val Acc: {val_acc}\n")

    if counter >= patience:
        print("Early stopping triggered")
        break