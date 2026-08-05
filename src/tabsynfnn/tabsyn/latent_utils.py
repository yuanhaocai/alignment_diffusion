# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import os
from collections import Counter
import sys
import json
import numpy as np
import torch
import torch.nn as nn


def get_input_train(args):
    dataname = args.dataname
    note = args.note

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'data/{dataname}'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/{note}'
    embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/{note}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_dir, ckpt_dir, info

 
@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, is_y_cond, y=None):
    """Split the generated synthetic data into numerical, categorical, and target columns.
    """
    device_ids = info['device_ids']
    device = device_ids[0] if device_ids else 'cpu'

    pre_decoder = info['pre_decoder'].to(device)
    
    if device_ids and info['vae_use_nn_parallel']:
        pre_decoder = nn.DataParallel(pre_decoder, device_ids=device_ids)
    x_hat_num, x_hat_cat = pre_decoder(torch.tensor(syn_data).to(device))

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim = -1))

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    if is_y_cond:
        if info['task_type'] == 'regression':
            sys.exit("Conditional generation for numerical y is not implemented yet.")
        else:
            y_dim = y.shape[1]
            y_arr = np.full((syn_cat.shape[0], y_dim), y)
            syn_cat_cmb = np.concatenate((y_arr, syn_cat), axis=1)
            
            try:
                syn_num = num_inverse(syn_num)
            except:
                pass # num_inverse is None, which indicates in the process, syn_num.shape[1] == 0
            try:
                syn_cat_cmb = cat_inverse(syn_cat_cmb)
            except:
                pass
            syn_target = syn_cat_cmb[:,:y_dim]
            syn_cat = syn_cat_cmb[:,y_dim:]

    else:
        try:
            syn_num = num_inverse(syn_num)
        except:
            pass
        try:
            syn_cat = cat_inverse(syn_cat)
        except:
            pass
        if info['task_type'] == 'regression':
            syn_target = syn_num[:, :1]
            syn_num = syn_num[:, 1:]
        else:
            syn_target = syn_cat[:, :1]
            syn_cat = syn_cat[:, 1:]

    return syn_num, syn_cat, syn_target


@torch.no_grad()
def split_generation_multimodal(syn_z_dat, syn_z_y, decoder_dat, decoder_y, num_inverse, cat_inverse, task_type, device_ids):
    """
    Process the generated latent embeddings and tranform back to the data space.
    """ 
    decoder_dat = nn.DataParallel(decoder_dat, device_ids=device_ids)
    x_hat_num, x_hat_cat = decoder_dat(syn_z_dat)

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim = -1))
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    if task_type in ['binclass', 'multiclass']:
        decoder_y = nn.DataParallel(decoder_y, device_ids=device_ids)
        y_hat = decoder_y(syn_z_y.unsqueeze(1))
        syn_y = []
        for pred in y_hat:
            syn_y.append(pred.argmax(dim = -1))
        syn_y = torch.stack(syn_y).t().cpu().numpy()
    else:
        syn_y = syn_z_y.cpu().numpy()


    if task_type in ['binclass', 'multiclass']:
        y_dim = syn_y.shape[1]
        syn_cat_cmb = np.concatenate((syn_y, syn_cat), axis=1)
        
        try:
            syn_num = num_inverse(syn_num)
        except:
            pass # num_inverse is None, which indicates in the process, syn_num.shape[1] == 0
        try:
            syn_cat_cmb = cat_inverse(syn_cat_cmb)
        except:
            pass
        syn_y = syn_cat_cmb[:,:y_dim]
        syn_cat = syn_cat_cmb[:,y_dim:]
    elif task_type == 'regression':
        y_dim = syn_y.shape[1]
        syn_num_cmb = np.concatenate((syn_y, syn_num), axis=1)
        try:
            syn_num_cmb = num_inverse(syn_num_cmb)
        except:
            pass
        try:
            syn_cat = cat_inverse(syn_cat)
        except:
            pass
        syn_y = syn_num_cmb[:, :y_dim]
        syn_num = syn_num_cmb[:, y_dim:]

    return syn_num, syn_cat, syn_y


@torch.no_grad()
def split_generation_dat2y(syn_z_y, decoder_y, task_type, num_dim, cat_dim, num_inverse, cat_inverse, device_ids,
                            return_proba=False):
    """
    Process the generated latent embeddings and tranform back to the data space
    for dat2y generation.

    Args:
        syn_z_y: torch.tensor, either generated VAE embeddings of y for classification or
                just the generated y for regression.
        num_dim, cat_dim: dimension of X_num, X_cat.
        return_proba: if True (classification only), additionally return the per-sample
                softmax class probabilities from the y decoder. This preserves the
                decoder's soft posterior instead of collapsing it via argmax, which
                lets the caller average decoded probabilities across repeats rather
                than hard-voting labels. The hard-label output (syn_y) is unchanged,
                so default behaviour (return_proba=False) is identical to before.

    Returns:
        syn_y                      if return_proba is False
        (syn_y, proba)             if return_proba is True, where proba has shape
                                   [batch_size, n_classes] for the (single) y column,
                                   columns ordered by the decoder's natural class order
                                   (0..K-1), matching infer_class_labels / predict_proba.
    """
    if task_type in ['binclass', 'multiclass']:
        if device_ids is not None:
            decoder_y = nn.DataParallel(decoder_y, device_ids=device_ids)
        y_hat = decoder_y(syn_z_y.unsqueeze(1))

        syn_y = []
        proba_cols = []
        for pred in y_hat:
            syn_y.append(pred.argmax(dim = -1))
            if return_proba:
                proba_cols.append(torch.softmax(pred, dim=-1))

        syn_y = torch.stack(syn_y).t().cpu().numpy()
        syn_cat = np.full((syn_y.shape[0], cat_dim), 0)

        y_dim = syn_y.shape[1]
        syn_cat_cmb = np.concatenate((syn_y, syn_cat), axis=1)
        syn_cat_cmb = cat_inverse(syn_cat_cmb)
        syn_y = syn_cat_cmb[:,:y_dim]

        if return_proba:
            # Only the y columns (the first y_dim categorical recon heads). For the
            # standard single-y case this is one column with shape [B, n_classes].
            proba = proba_cols[0].detach().cpu().numpy() if len(proba_cols) == 1 \
                else [c.detach().cpu().numpy() for c in proba_cols[:y_dim]]
            return syn_y, proba
    elif task_type == 'regression':
        syn_z_y = syn_z_y.cpu().numpy()
        syn_num = np.full((syn_z_y.shape[0], num_dim), 0)
        y_dim = syn_z_y.shape[1]
        syn_num_cmb = np.concatenate((syn_z_y, syn_num), axis=1)
        syn_num_cmb = num_inverse(syn_num_cmb)
        syn_y = syn_num_cmb[:,:y_dim]
        if return_proba:
            return syn_y, None

    return syn_y


@torch.no_grad()
def decode_nearest_response_prototype(
    syn_z_y,
    prototypes,
    class_indices,
    cat_dim,
    cat_inverse,
    return_proba=False,
):
    """Decode generated categorical-response embeddings by nearest prototype.

    ``prototypes`` contains one learned response-encoder vector per encoded
    class.  Hard class probabilities reported by the paper are still obtained
    from repeated nearest-prototype labels; the optional soft probabilities are
    a distance-based diagnostic only.
    """
    if syn_z_y.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("syn_z_y and prototypes must both be two-dimensional")
    if syn_z_y.shape[1] != prototypes.shape[1]:
        raise ValueError(
            f"embedding width mismatch: samples={syn_z_y.shape[1]} "
            f"prototypes={prototypes.shape[1]}"
        )
    if prototypes.shape[0] != len(class_indices):
        raise ValueError("class_indices must contain one value per prototype")

    distances = torch.cdist(syn_z_y, prototypes)
    nearest = distances.argmin(dim=1)
    class_indices = torch.as_tensor(
        class_indices, device=nearest.device, dtype=torch.long
    )
    encoded = class_indices[nearest].cpu().numpy().reshape(-1, 1)
    categorical_placeholders = np.zeros((encoded.shape[0], cat_dim), dtype=int)
    combined = np.concatenate((encoded, categorical_placeholders), axis=1)
    decoded = cat_inverse(combined)[:, :1]

    if return_proba:
        diagnostic_proba = torch.softmax(-distances, dim=1).cpu().numpy()
        return decoded, diagnostic_proba
    return decoded


def regression_average(samples):
    """
    Computes the average prediction across repeats for regression.

    Args:
        samples: np.ndarray of shape [num_samples, test_size, 1]

    Returns:
        np.ndarray of shape [test_size, 1] with the mean value per test sample
    """
    avg = samples.mean(axis=0)  # shape [test_size, 1]
    return avg


def majority_vote(samples):
    """
    Note that for ties, the label that appears first in the samples is chosen. May be improved.
    
    Args:
        samples: np.ndarray of shape [num_samples, test_size, 1]
    
    Returns: 
        np.ndarray of shape [test_size, 1] with the majority label
    """
    # Convert to shape [test_size, num_samples] for easier voting
    labels = samples.squeeze(-1).T  # [test_size, num_samples]
    voted = []
    for row in labels:
        # row: [num_samples] of str
        count = Counter(row)
        maj = count.most_common(1)[0][0]
        voted.append(maj)
    return np.array(voted).reshape(-1, 1)


# def compute_proba(samples):    
#     """
#     Compute predicted probability for binary classification.

#     Args:
#         samples: same as in ``majority_vote``.

#     Returns:
#         np.ndarray: Probabilities for the positive class (shape: [test_size, 1])
#     """
#     all_samples_arr = samples.squeeze(-1).T  # [test_size, num_repeats]

#     # Compute probability as the fraction of ones (positive class) for each example
#     prob = np.mean(all_samples_arr == '1', axis=1)
#     # Return as [test_size, 1]
#     return prob.reshape(-1, 1)


def compute_proba(samples, class_labels=None, positive_label="1"):
    """
    Compute predicted class probabilities from repeated generated labels.

    Args:
        samples: array-like repeated samples with shape [num_repeats, test_size, 1]
            or [num_repeats, test_size].
        class_labels: ordered class labels for multiclass probability columns. When
            omitted, the binary behavior is kept and only positive-class probability
            is returned.
        positive_label: positive class label for binary classification.

    Returns:
        Binary classification: np.ndarray with shape [test_size, 1], containing the
        positive-class probability, matching the previous implementation.

        Multiclass classification: np.ndarray with shape [test_size, n_classes],
        where column j is P(y == class_labels[j]), matching sklearn predict_proba.
    """
    samples = np.asarray(samples)
    if samples.ndim == 3 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)
    if samples.ndim != 2:
        raise ValueError(
            "samples must have shape [num_repeats, test_size, 1] or "
            f"[num_repeats, test_size]; got {samples.shape}"
        )

    all_samples_arr = samples.T  # [test_size, num_repeats]
    if class_labels is None or len(class_labels) == 2:
        if class_labels is not None:
            positive_label = class_labels[1]
        prob = np.mean(all_samples_arr == positive_label, axis=1)
        return prob.reshape(-1, 1)

    class_labels = np.asarray(class_labels)
    return np.column_stack(
        [np.mean(all_samples_arr == class_label, axis=1) for class_label in class_labels]
    )
