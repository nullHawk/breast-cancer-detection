# preprocess_data.py
import pandas as pd

def load_data():
    # Load the datasets
    df_mega_train = pd.read_csv('/path/to/mass_case_description_train_set.csv')
    df_mega_test = pd.read_csv('/path/to/mass_case_description_test_set.csv')
    df_dicom = pd.read_csv('/path/to/dicom_info.csv')

    return df_mega_train, df_mega_test, df_dicom

def preprocess_image_paths(df_dicom):
    imdir = '/data/archive/jpeg'

    # Fix image paths
    cropped_images = df_dicom[df_dicom.SeriesDescription == 'cropped images'].image_path.replace('CBIS-DDSM/jpeg', imdir, regex=True)
    full_mammo = df_dicom[df_dicom.SeriesDescription == 'full mammogram images'].image_path.replace('CBIS-DDSM/jpeg', imdir, regex=True)
    roi_img = df_dicom[df_dicom.SeriesDescription == 'ROI mask images'].image_path.replace('CBIS-DDSM/jpeg', imdir, regex=True)

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

def preprocess_data(df_mega_train, df_mega_test, full_mammo_dict, cropped_images_dict):
    # Rename columns
    df_mega_train = df_mega_train.rename(columns={'left or right breast': 'left_or_right_breast', 'image view': 'image_view', 'abnormality id': 'abnormality_id', 'abnormality type': 'abnormality_type', 'mass shape': 'mass_shape', 'mass margins': 'mass_margins', 'image file path': 'image_file_path', 'cropped image file path': 'cropped_image_file_path', 'ROI mask file path': 'ROI_mask_file_path'})

    # Fix image paths
    fix_image_path(df_mega_train, full_mammo_dict, cropped_images_dict)
    fix_image_path(df_mega_test, full_mammo_dict, cropped_images_dict)

    # Fill missing values
    df_mega_train['mass_shape'] = df_mega_train['mass_shape'].fillna(method='bfill')
    df_mega_train['mass_margins'] = df_mega_train['mass_margins'].fillna(method='bfill')

    return df_mega_train, df_mega_test

def load_and_preprocess_data():
    df_mega_train, df_mega_test, df_dicom = load_data()
    full_mammo_dict, cropped_images_dict = preprocess_image_paths(df_dicom)
    return preprocess_data(df_mega_train, df_mega_test, full_mammo_dict, cropped_images_dict)
