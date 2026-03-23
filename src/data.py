from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_dataloaders(train_dir, val_dir, batch_size=config["data"]["batch_size"]):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader