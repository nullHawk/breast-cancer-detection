from preprocess_dataset import preprocess_data, get_dataloader
from train_eval import k_fold_cross_validation

# Define paths for input/output directories
input_dir = '/data/archive/jpeg'
output_dir = 'path_to_save_pngs'
data_dir = output_dir


preprocess_data(input_dir, output_dir)
train_loader, val_loader = get_dataloader(data_dir, batch_size=32)
k_fold_cross_validation(train_loader.dataset, k=5)
