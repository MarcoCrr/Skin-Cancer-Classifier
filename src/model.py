import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, freeze_backbone=True):
    """
    Initialize a ResNet18 model for classification.

    This function loads a pretrained ResNet18 model and adapts its final
    fully connected layer to match the desired number of output classes.
    Optionally, the backbone (all layers except the final classifier) can
    be frozen for transfer learning.

    Args:
        num_classes (int, optional):
            Number of output classes. Default is 2 (binary classification).

        freeze_backbone (bool, optional):
            If True, all pretrained layers are frozen (i.e., their weights
            are not updated during training). Only the final classification
            layer remains trainable. Default is True.

    Returns:
        torch.nn.Module:
            A ResNet18 model with a modified final layer.

    Notes:
        - Uses pretrained weights ("IMAGENET1K_V1").
        - When `freeze_backbone=True`, only `model.fc` is trainable.
        - Suitable for transfer learning on SMALL datasets.
    """

    model = models.resnet18(weights="IMAGENET1K_V1")
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model