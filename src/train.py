import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import argparse
from pathlib import Path

from data import get_dataloaders
from model import get_model
from trainer import train_one_epoch, evaluate, should_stop_early



def train(config):

    device = config["system"]["device"] if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_dataloaders(
        config["data"]["train_dir"],
        config["data"]["val_dir"],
        batch_size=config["data"]["batch_size"]
    )

    model = get_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]

    best_acc = 0
    counter = 0

    for epoch in range(epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        best_model, counter = should_stop_early(val_acc, best_acc, counter)

        if best_model:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pth")

        if counter >= patience:
            print("Early stopping triggered")
            break

    return best_acc


def main(args):
    train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config/input file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open("logs/config_used.json", "w") as f: #TODO: add date or whatever to distinguish the files
        json.dump(config, f, indent=4)
    main(args)