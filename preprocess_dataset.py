import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data():
    # Load the datasets
    df_train = pd.read_csv('/path/to/mass_case_description_train_set.csv')
    df_test = pd.read_csv('/path/to/mass_case_description_test_set.csv')
    df_dicom = pd.read_csv('/path/to/dicom_info.csv')

    return df_train, df_test, df_dicom

def preprocess_image_paths(df_dicom):
    imdir = '/data/archive/jpeg'

    # Fix image paths
    cropped_images = df_dicom[df_dicom.SeriesDescription == 'cropped images'].image_path.replace('CBIS-DDSM/jpeg', imdir, regex=True)
    full_mammo = df_dicom[df_dicom.SeriesDescription == 'full mammogram images'].image_path.replace('CBIS-DDSM/jpeg', imdir, regex=True)

    # Organize image paths into dictionaries
    full_mammo_dict = {dicom.split("/")[4]: dicom for dicom in full_mammo}
    cropped_images_dict = {dicom.split("/")[4]: dicom for dicom in cropped_images}

    return full_mammo_dict, cropped_images_dict

def fix_image_path(df, full_mammo_dict, cropped_images_dict):
    """Fix image paths in the dataframe using dictionaries."""
    for index, img in df.iterrows():
        img_name_full = img['image_file_path'].split("/")[2]
        img_name_cropped = img['cropped_image_file_path'].split("/")[2]
        df.at[index, 'image_file_path'] = full_mammo_dict[img_name_full]
        df.at[index, 'cropped_image_file_path'] = cropped_images_dict[img_name_cropped]

def preprocess_data(df_train, df_test, full_mammo_dict, cropped_images_dict):
    # Fix image paths
    fix_image_path(df_train, full_mammo_dict, cropped_images_dict)
    fix_image_path(df_test, full_mammo_dict, cropped_images_dict)

    # Return preprocessed DataFrames
    return df_train, df_test

def prepare_dataloader(df, batch_size=32):
    """
    Convert the dataframe into a PyTorch DataLoader.
    
    Args:
        df (pd.DataFrame): The dataframe with features and labels.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        DataLoader: DataLoader for the input dataframe.
    """
    # Assuming you have your image tensors and labels ready for PyTorch
    # Here you would load images based on df['image_file_path'] and normalize them.
    # For demonstration, we'll use random tensors.

    # Create random feature tensors (replace this with actual image loading)
    X = torch.randn(len(df), 3, 224, 224)  # Example: 3-channel images (RGB), size 224x224
    y = torch.tensor(df['pathology'].values, dtype=torch.long)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def load_and_preprocess_data():
    df_train, df_test, df_dicom = load_data()
    full_mammo_dict, cropped_images_dict = preprocess_image_paths(df_dicom)
    df_train, df_test = preprocess_data(df_train, df_test, full_mammo_dict, cropped_images_dict)

    # Create DataLoaders
    train_loader = prepare_dataloader(df_train)
    test_loader = prepare_dataloader(df_test)

    return train_loader, test_loader
