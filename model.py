import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_resnet(num_classes=2):
    """
    Load a pretrained ResNet model and modify the final layer for binary classification.

    Args:
        num_classes (int): Number of output classes (2 for binary classification).
    
    Returns:
        model (torch.nn.Module): The modified ResNet model.
    """
    # Load the pretrained ResNet model
    model = models.resnet18(pretrained=True)
    
    # Freeze the lower layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the final fully connected layer to have the correct number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


def get_device():
    """
    Return the device to use (GPU if available, otherwise CPU).
    
    Returns:
        device (torch.device): The device to use.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
