import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, freeze_backbone=True):
    model = models.resnet18(pretrained=True)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model