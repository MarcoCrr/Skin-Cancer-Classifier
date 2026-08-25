from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_dataloaders(train_dir,
                    val_dir,
                    batch_size=config["data"]["batch_size"],
                    num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, val_loader