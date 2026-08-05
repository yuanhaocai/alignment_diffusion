"""
Functions and classes for data.
"""
import os
from typing import Literal

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import torch
from torch.utils.data import Dataset


class Dataset_text_tabular(Dataset):
    """
    torch.Dataset subclass for text + tabular data. Combines train and test splits into a single dataset.

    Args:
        text_path: shall be a directory containing 'train/' and 'val/' subdirectories, containing
                    encoded embeddings from the CLIP text encoder.
    """
    def __init__(
            self, 
            text_path, 
            train_z_dat_path,
            y_path,
            y_type: Literal['continuous', 'categorical'],
            split: Literal['train', 'test', 'val'],
        ):
        assert split in ["train", "test", "val"]
        self.text_path = text_path
        self.split = split

        train_z_dat = np.load(os.path.join(train_z_dat_path, 'train_z_dat.npy'))
        split_z_dat = np.load(os.path.join(train_z_dat_path, f'{split}_z_dat.npy'))
        if y_type == 'continuous':
            y_split = np.load(os.path.join(y_path, f'y_{split}.npy')).squeeze().astype(np.float32)
            y_dtype = torch.float
        elif y_type == 'categorical':
            y_split = np.load(os.path.join(y_path, f'y_{split}.npy')).squeeze().astype(np.int64)  # shape [N, 1], strings
            y_dtype = torch.long
        else:
            raise ValueError(f"Unsupported y_type: {y_type}")

        train_z_dat = torch.tensor(train_z_dat).float()
        mean, std = train_z_dat.mean(0), train_z_dat.std(0)
        split_z_dat = torch.tensor(split_z_dat).float()
        self.z_dat = (split_z_dat - mean) / (std + 1e-6)
        self.y = torch.tensor(y_split, dtype=y_dtype)

    def __len__(self):
        return self.z_dat.shape[0]

    def __getitem__(self, idx):
        """
        Output shape: 
        tab_dat: [B, latent_dim]; text: [B, 77, 768]; y: [B]
        """
        text_emd = np.load(os.path.join(self.text_path, self.split, f'{idx}.npy')).squeeze()

        return self.z_dat[idx,], self.y[idx], text_emd
    

def load_data(
        data_path=None, 
        train_z_dat_path=None,
        predictor_to_use: Literal['vae', 'original']='original',
        task: Literal['cls', 'reg']='cls',
        val_set=False
    ):
    """
    Load data for baseline model fitting or tuning.
    """
    ## predictors
    if predictor_to_use == "vae":
        train_z = np.load(os.path.join(train_z_dat_path, 'train_z_dat.npy'))
        test_z = np.load(os.path.join(train_z_dat_path, 'test_z_dat.npy'))
        mean, std = train_z.mean(0), train_z.std(0)
        train_z_normalized = (train_z - mean) / (std + 1e-6)     
        test_z_normalized = (test_z - mean) / (std + 1e-6)

        # Standardize based on train
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_z_normalized)
        X_test = scaler.transform(test_z_normalized)

    elif predictor_to_use == "original":
        num_train = np.load(os.path.join(data_path, 'X_num_train.npy'))
        cat_train = np.load(os.path.join(data_path, 'X_cat_train.npy'))
        num_test = np.load(os.path.join(data_path, 'X_num_test.npy'))
        cat_test = np.load(os.path.join(data_path, 'X_cat_test.npy'))
        if val_set:
            num_val = np.load(os.path.join(data_path, 'X_num_val.npy'))
            cat_val = np.load(os.path.join(data_path, 'X_cat_val.npy'))

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), slice(0, num_train.shape[1])),        # scale numeric
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), 
                # ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
                slice(num_train.shape[1], num_train.shape[1] + cat_train.shape[1])) # OHE categorical
            ]
        )
        # Stack numerical and categorical for the ColumnTransformer
        X_train_all = np.hstack([num_train, cat_train])
        X_test_all = np.hstack([num_test, cat_test])
        # Fit on train, transform both train and test
        X_train = preprocessor.fit_transform(X_train_all)
        X_test = preprocessor.transform(X_test_all)
        if val_set:
            X_val_all = np.hstack([num_val, cat_val])
            X_val = preprocessor.transform(X_val_all)

    ## response
    y_train = np.load(os.path.join(data_path, 'y_train.npy')) #.squeeze()
    y_test = np.load(os.path.join(data_path, 'y_test.npy')) #.squeeze()
    if val_set:
        y_val = np.load(os.path.join(data_path, 'y_val.npy')) #.squeeze()

    if task == 'cls':
        if not np.issubdtype(y_train.dtype, np.number):
            print("Using LabelEncoder for non-numeric response")
            le = LabelEncoder()
            y_train = le.fit_transform(y_train.squeeze())
            y_test = le.transform(y_test.squeeze())
            if val_set:
                y_val = le.transform(y_val.squeeze())
        else:
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()
            if val_set:
                y_val = y_val.squeeze()
    elif task == 'reg':
        print("Using StandardScaler to normalize continuous response")
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train).squeeze()
        y_test = scaler.transform(y_test).squeeze()
        if val_set:
            y_val = scaler.transform(y_val).squeeze()
        
    if not val_set:
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test, X_val, y_train, y_test, y_val
