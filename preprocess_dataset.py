import os
import pydicom
import cv2
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def dicom_to_image(dicom_file_path, output_path):
    """
    Convert a DICOM file to a PNG image for easier processing.

    Args:
        dicom_file_path (str): Path to the DICOM file.
        output_path (str): Path to save the output PNG image.
    """
    dicom = pydicom.dcmread(dicom_file_path)
    img = dicom.pixel_array
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype('uint8')
    cv2.imwrite(output_path, img)


def preprocess_data(input_dir, output_dir):
    """
    Preprocess all DICOM files in the input directory and convert them to PNG.

    Args:
        input_dir (str): Directory containing the DICOM files.
        output_dir (str): Directory to save the converted PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.dcm'):
                dicom_file_path = os.path.join(subdir, file)
                output_file = os.path.join(output_dir, file.replace('.dcm', '.png'))
                dicom_to_image(dicom_file_path, output_file)


def get_dataloader(data_dir, batch_size=32, train_split=0.8):
    """
    Prepare the dataset by applying data augmentations and splitting it into train/validation sets.

    Args:
        data_dir (str): Path to the directory containing the images.
        batch_size (int): Batch size for the dataloaders.
        train_split (float): Ratio for the train-validation split.
    
    Returns:
        train_loader, val_loader (DataLoader, DataLoader): DataLoader objects for training and validation.
    """
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = ImageFolder(root=data_dir, transform=data_transforms)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
