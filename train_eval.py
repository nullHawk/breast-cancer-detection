import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from model import get_pretrained_resnet, get_device


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to train the model on (CPU or GPU).
    
    Returns:
        float: Training loss.
    """
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_model(model, val_loader, device):
    """
    Evaluate the model on the validation data and calculate performance metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): Device to evaluate the model on (CPU or GPU).
    
    Returns:
        dict: Dictionary containing evaluation metrics (AUC-ROC, sensitivity, specificity).
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate AUC-ROC
    auc = roc_auc_score(all_labels, all_preds)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {
        'AUC-ROC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Confusion Matrix': cm
    }


def k_fold_cross_validation(dataset, k=5):
    """
    Perform k-fold cross-validation on the dataset.

    Args:
        dataset (Dataset): The dataset to split for cross-validation.
        k (int): Number of folds (default 5).
    """
    kf = KFold(n_splits=k)
    fold = 1
    device = get_device()
    
    for train_idx, val_idx in kf.split(dataset):
        print(f'Fold {fold}')

        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

        # Initialize the model, loss, and optimizer
        model = get_pretrained_resnet(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

        # Train the model
        for epoch in range(10):  # You can adjust the number of epochs
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}')

        # Evaluate the model
        metrics = evaluate_model(model, val_loader, device)
        print(f'Fold {fold} Metrics: {metrics}')
        
        fold += 1
