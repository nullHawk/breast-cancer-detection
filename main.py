from preprocess_dataset import load_and_preprocess_data
from train_eval import train_model, evaluate_model
from model import get_pretrained_resnet, get_device

def main():
    # Load and preprocess the data
    train_loader, test_loader = load_and_preprocess_data()

    # Initialize the ResNet model
    device = get_device()
    model = get_pretrained_resnet(num_classes=2)

    # Train the model
    train_model(model, train_loader, device, num_epochs=10)

    # Evaluate the model
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
