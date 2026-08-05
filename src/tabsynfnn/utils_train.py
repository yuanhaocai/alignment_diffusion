# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import os

import numpy as np
import torch

from .utils.data import Transformations, read_pure_data, load_json, Dataset, TaskType, transform_dataset, change_val
from .utils.util import get_categories


def compute_norms(model_params, grad_scale=1.0):
    """A helper function to compute the model parameter norm and gradient norm.
    Borrowed from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/fp16_util.py#L217
    
    Args:
        model_params: shall be `list(model.parameters())` with `model` being a nn.Module instance.
    """
    grad_norm = 0.0
    param_norm = 0.0
    for p in model_params:
        with torch.no_grad():
            param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
            if p.grad is not None:
                grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
    return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __getitem__(self, index):
        return (self.X_num[index], self.X_cat[index], self.y[index])

    def __len__(self):
        return self.X_num.shape[0]
    

class LatentDataset(Dataset):
    """Child class of Torch dataset for providing a latent embedding array and a label
    or two latent embedding arrays.
    """
    def __init__(self, z, y):
        self.z = z
        self.y = y
        assert len(self.y) == len(self.z)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
         return self.z[idx,], self.y[idx,]


def preprocess(dataset_path, task_type, is_y_cond=False, inverse=False, cat_encoding=None):
    """Preprocess data. 
    
    i=If `is_y_cond=False`, will concatenate y with X_num or X_cat depending on `task_type` and return 
    randomly generated y. Otherwise will not do the concatenation and return the true ys.

    Return:
        categories: get_categories(X_train_cat).
        d_numerical: X_num_train.shape[1].
    """
    T_dict = {}
    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = Transformations(**T_dict)

    dataset = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        to_change_val = False,
        concat = True
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        try:
            X_train_cat, X_test_cat = X_cat['train'], X_cat['test']
            data_size = (X_train_cat.shape[0], X_test_cat.shape[0])
        except:
            print('No categorical data')

        try:
            X_train_num, X_test_num = X_num['train'], X_num['test']
            data_size = (X_train_num.shape[0], X_test_num.shape[0])
            if X_cat is None:
                X_train_cat, X_test_cat = np.empty((data_size[0],0)), np.empty((data_size[1],0))
        except:
            print('No numerical data')
            X_train_num, X_test_num = np.empty((data_size[0],0)), np.empty((data_size[1],0))

        try:
            X_val_cat = X_cat['val']
        except:
            X_val_cat = np.empty((data_size[1], X_test_cat.shape[1]))
        try:
            X_val_num = X_num['val']
        except:
            X_val_num = np.empty((data_size[1], X_test_num.shape[1]))

        if is_y_cond:
            print('`is_y_cond` is `True` in `preprocess`.')
            dim_cond = dataset.y['train'].shape[1]
            if task_type == 'regression':
                print('y is numerical.')
                y_train, y_test, y_val = X_train_num[:,:dim_cond], X_test_num[:,:dim_cond], X_val_num[:,:dim_cond]
                X_train_num, X_test_num, X_val_num = X_train_num[:,dim_cond:], X_test_num[:,dim_cond:], X_val_num[:,dim_cond:]
            else:
                print('y is categorical.')
                y_train, y_test, y_val = X_train_cat[:,:dim_cond], X_test_cat[:,:dim_cond], X_val_cat[:,:dim_cond]
                X_train_cat, X_test_cat, X_val_cat = X_train_cat[:,dim_cond:], X_test_cat[:,dim_cond:], X_val_cat[:,dim_cond:]
        else:
            # y_train, y_test = X_train_num[:,:1], X_test_num[:,:1] # just place holders. This will be provided to the mlp network but not used anyway
            y_train, y_test = np.ones((data_size[0], 1)), np.ones((data_size[1], 1))

        categories = get_categories(X_train_cat) # get_categories() does detect `None`
        d_numerical = X_train_num.shape[1] # if X_num is not None else 0

        X_num = (X_train_num, X_test_num, X_val_num) # if X_num is not None else (None, None)
        X_cat = (X_train_cat, X_test_cat, X_val_cat) # if X_cat is not None else (None, None)
        y = (y_train, y_test, y_val)

        if inverse:
            try:
                num_inverse = dataset.num_transform.inverse_transform
            except:
                num_inverse = None
            try:
                cat_inverse = dataset.cat_transform.inverse_transform
            except:
                cat_inverse = None
            return X_num, X_cat, y, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, y, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


def concat_y_to_X(X, y):
    num_sam = len(y)
    if X is None:
        return y.reshape(num_sam, -1)
    return np.concatenate([y.reshape(num_sam, -1), X], axis=1)


def make_dataset(
    data_path: str,
    T: Transformations,
    task_type,
    to_change_val: bool,
    concat=True,
):

    # classification
    if task_type == 'binclass' or task_type == 'multiclass' or task_type == 'cat_cond':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test', 'val']:
            try:
                X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
                if X_num is not None:
                    X_num[split] = X_num_t
                if X_cat is not None:
                    if concat:
                        X_cat_t = concat_y_to_X(X_cat_t, y_t)
                    X_cat[split] = X_cat_t  
                if y is not None:
                    y[split] = y_t
            except:  # in case there is no validationn set
                print(f"inside function `make_dataset`: split {split} not found")
                pass
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test', 'val']:
            try:
                X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)

                if X_num is not None:
                    if concat:
                        X_num_t = concat_y_to_X(X_num_t, y_t)
                    X_num[split] = X_num_t
                if X_cat is not None:
                    X_cat[split] = X_cat_t
                if y is not None:
                    y[split] = y_t
            except:
                print(f"inside function `make_dataset`: split {split} not found")
                pass

    # info = load_json(os.path.join(data_path, 'info.json'))

    D = Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=TaskType(task_type),
        n_classes=None
    )

    if to_change_val:
        D = change_val(D)

    return transform_dataset(D, T, None)
