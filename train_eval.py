import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

def train_model(model, train_loader, device, num_epochs=10):
    """
    Train the model using the provided data loader.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        device (torch.device): Device to run the model on (CPU or GPU).
        num_epochs (int): Number of training epochs.
    """
    # Move model to the appropriate device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set.

    Args:
        model (torch.nn.Module): The neural network model.
        test_loader (DataLoader): DataLoader for the test data.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    # Move model to evaluation mode
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    print(classification_report(true_labels, pred_labels))
