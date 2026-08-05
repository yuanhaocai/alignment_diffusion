import warnings
# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import os
import sys
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ...utils.util import get_categories
from .model_fnn import Model_VAE_fnn, Encoder_model_fnn, Decoder_model_fnn
from .model_multimodal import VAE_caty, Encoder_caty, Decoder_caty
from ...utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')


def _compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    """
    Args:
        X_num, Recon_X_num: could have 0 columns. The corresponding mse_loss will be zero.
        X_cat, Recon_X_cat: have to have > 0 columns for now.
    """
    
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss = 0; acc = 0; total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    
    return mse_loss, ce_loss, loss_kld, acc


def _compute_loss_caty(y, recon_y):
    """
    Helper function for computing loss for fitting VAE on a categorical y.

    Args:
        y: shall be of shape (batch, dim_y, *), which is at least 2-dimensional.
    """
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss = 0; acc = 0; total_num = 0

    for idx, y_col in enumerate(recon_y):
        ce_loss += ce_loss_fn(y_col, y[:, idx])
        y_col_hat = y_col.argmax(dim = -1)
        acc += (y_col_hat == y[:,idx]).float().sum()
        total_num += y_col_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    
    return ce_loss, acc


def vae_train_unitabsyn(data_path, vae_path, latent_dim=None, d_token=4, task_type='binclass', train_target=None, device_ids=None,
                        epochs=4000, batch_size=4096, lr=1e-3, max_beta=1e-2, min_beta=1e-5, lambd=0.7):
    """
    Train the VAE, for use in uni-tabsyn.

    Depending on `train_target`, will save hyperparameters and weights with different
    suffixes.

    Args:
        latent_dim: the dimension of the latent space of the VAE. Only used for `target`=`dat`.
        d_token: the dimension the feature tokenizer will project the vector to. Default is 4. 
                Refer to the tabsyn paper for more details.
        task_type: Set to `regression` to indicate that y_train.npy, y_test.npy contain numerical data.
                   Set to `binclass` or `multiclass` to indicate they are categorical.
                   When is_y_cond is False, `regression`, `binclass`/`multiclass` are both good. y_train.npy, y_test.npy will be concatenated to X_cat or X_num.
                   When is_y_cond is True, only `binclass`/`multiclass` is available for now.
        train_target: choose from 'y', 'dat', 'cat'. indicating training a VAE for y; for the 
                        X_cat and X_num; for the X_cat.
        device_ids: the indices of the GPUs to use.
    """
    assert train_target in ['dat', 'y', 'cat']
    device = device_ids[0]

    if not os.path.exists(vae_path):
        os.makedirs(vae_path)

    X_num, X_cat, y, categories, d_numerical = preprocess(data_path, task_type=task_type, is_y_cond=True)
    if train_target == 'y':
        categories = get_categories(y[0])

    if not os.path.exists(os.path.join(data_path, "y_val.npy")):
        raise FileNotFoundError("VAE training requires y_val.npy")
    eval_index = 2
    X_train_num, X_test_num = X_num[0], X_num[eval_index]
    X_train_cat, X_test_cat = X_cat[0], X_cat[eval_index]
    y_train, y_test = y[0], y[eval_index]

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)
    y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)

    train_dataset = TabularDataset(X_train_num.float(), X_train_cat, y_train)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)
    y_test = y_test.to(device)

    if train_target == 'dat':
        seq_len = X_train_num.shape[1] + X_train_cat.shape[1]
    elif train_target == 'y':
        seq_len = y_train.shape[1]
    elif train_target == 'cat':
        seq_len = X_train_cat.shape[1]
    else:
        sys.exit("Not implemented train_target.")
    
    hypar_file_dir = os.path.join(vae_path, f'hyperparameters_{train_target}.txt')
    with open(hypar_file_dir, 'w') as file:
        file.write(f"vae_epochs: {epochs}\n")
        file.write(f"vae_lr: {lr}\n")
        file.write(f"latent_dim: {latent_dim if train_target == 'dat' else None}\n")
        file.write(f"batch_size: {batch_size}\n")
        file.write(f"d_token: {d_token}\n")
        file.write(f"seq_len: {seq_len}\n")
        file.write(f"max_beta: {max_beta}\n")
        file.write(f"min_beta: {min_beta}\n")
        file.write(f"lambd: {lambd}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    if train_target == 'dat':
        model = Model_VAE_fnn(d_numerical, categories, d_token, seq_len, latent_dim, bias=True).to(device)
        model = nn.DataParallel(model, device_ids=device_ids)
        pre_encoder = Encoder_model_fnn(d_numerical, categories, d_token, seq_len, latent_dim).to(device)
        pre_decoder = Decoder_model_fnn(d_numerical, categories, d_token, seq_len, latent_dim).to(device)
    elif train_target in ['y', 'cat']:
        model = VAE_caty(categories, d_token, bias=True).to(device)
        model = nn.DataParallel(model, device_ids=device_ids)
        pre_encoder = Encoder_caty(categories, d_token).to(device)
        pre_decoder = Decoder_caty(categories, d_token).to(device)
    
    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)

    best_train_loss = float('inf')
    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    if train_target == 'dat':
        loss_history = pd.DataFrame(columns=['epoch', 'num_loss', 'cat_loss', 'train_loss', 'kl_loss'])
    elif train_target in ['y', 'cat']:
        loss_history = pd.DataFrame(columns=['epoch', 'train_ce', 'val_ce', 'train_acc', 'val_acc'])

    start_time = time.time()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")

        curr_loss_multi = 0.0; curr_loss_gauss = 0.0; curr_loss_kl = 0.0
        curr_acc = 0.0
        curr_count = 0

        if train_target == 'dat':
            for batch_num, batch_cat, batch_y in pbar:
                model.train()
                optimizer.zero_grad()

                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device)
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
                loss_mse, loss_ce, loss_kld, train_acc = _compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
                loss = loss_mse + loss_ce + beta * loss_kld
                loss.backward()
                optimizer.step()

                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_multi += loss_ce.item() * batch_length
                curr_loss_gauss += loss_mse.item() * batch_length
                curr_loss_kl += loss_kld.item() * batch_length

            num_loss = curr_loss_gauss / curr_count
            cat_loss = curr_loss_multi / curr_count
            kl_loss = curr_loss_kl / curr_count
            train_loss = num_loss + cat_loss

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), os.path.join(vae_path, f'model_{train_target}.pt'))
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd

        elif train_target in ['y', 'cat']:
            for batch_num, batch_cat, batch_y in pbar:
                if train_target == 'y':
                    batch = batch_y
                elif train_target == 'cat':
                    batch = batch_cat
                model.train()
                optimizer.zero_grad()

                batch = batch.to(device)
                recon_caty, _ = model(batch)
                loss_ce, train_acc = _compute_loss_caty(batch, recon_caty)
                loss = loss_ce
                loss.backward()
                optimizer.step()

                batch_length = batch.shape[0]
                curr_count += batch_length
                curr_loss_multi += loss_ce.item() * batch_length
                curr_acc += train_acc.item() * batch_length

            cat_loss = curr_loss_multi / curr_count
            cat_acc = curr_acc / curr_count   
            train_loss = cat_loss
            
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), os.path.join(vae_path, f'model_{train_target}.pt'))

        # scheduler.step(train_loss)
            
        model.eval()
        with torch.no_grad():
            if train_target == 'dat':
                Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)
                val_mse_loss, val_ce_loss, val_kl_loss, val_acc = _compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
                val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    current_lr = new_lr
                    print(f"Learning rate updated: {current_lr}")
                print('epoch: {}, beta={:.6f}, Train MSE:{:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch+1, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item()))
                loss_history.loc[len(loss_history)] = [epoch+1, np.round(num_loss, 6), np.round(cat_loss, 6), np.round(train_loss, 6), np.round(kl_loss, 6)]
            elif train_target in ['y', 'cat']:
                if train_target == 'y':
                    recon_caty, _ = model(y_test)
                    val_ce_loss, val_acc = _compute_loss_caty(y_test, recon_caty)
                elif train_target == 'cat':
                    recon_cat, _ = model(X_test_cat)
                    val_ce_loss, val_acc = _compute_loss_caty(X_test_cat, recon_cat)
                val_loss = val_ce_loss.item()
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    current_lr = new_lr
                    print(f"Learning rate updated: {current_lr}")
                print('epoch: {}, Train CE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch+1, cat_loss, val_ce_loss.item(), cat_acc, val_acc.item()))
                loss_history.loc[len(loss_history)] = [epoch+1, np.round(cat_loss, 6), np.round(val_loss, 6), np.round(cat_acc, 6), np.round(val_acc.item(), 6)]

        # gc.collect()  # gc after each epoch


    end_time = time.time()
    loss_history.to_csv(os.path.join(vae_path, f'loss_{train_target}.csv'), index=False)

    with open(hypar_file_dir, 'a') as file:
        file.write(f"Time used: {end_time - start_time} seconds\n") 

    with torch.no_grad():
        pre_encoder.load_weights(model.module)
        pre_decoder.load_weights(model.module)

        torch.save(pre_encoder.state_dict(), os.path.join(vae_path, f'encoder_{train_target}.pt'))
        torch.save(pre_decoder.state_dict(), os.path.join(vae_path, f'decoder_{train_target}.pt'))
        
        if train_target == 'dat':
            X_train_num, X_train_cat = X_train_num.to(device), X_train_cat.to(device)
            X_test_num, X_test_cat = X_test_num.to(device), X_test_cat.to(device)
            pre_encoder = nn.DataParallel(pre_encoder, device_ids=device_ids)
            train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()
            np.save(os.path.join(vae_path, f'train_z_{train_target}.npy'), train_z)
            test_z = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()
            np.save(os.path.join(vae_path, f'test_z_{train_target}.npy'), test_z)
        elif train_target == 'y':
            y_train = y_train.to(device)
            pre_encoder = nn.DataParallel(pre_encoder, device_ids=device_ids)
            train_z = pre_encoder(y_train).detach().cpu().numpy()
            np.save(os.path.join(vae_path, f'train_z_{train_target}.npy'), train_z)
            test_z = pre_encoder(y_test).detach().cpu().numpy()
            np.save(os.path.join(vae_path, f'test_z_{train_target}.npy'), test_z)
        elif train_target == 'cat':
            X_train_cat, X_test_cat = X_train_cat.to(device), X_test_cat.to(device)
            pre_encoder = nn.DataParallel(pre_encoder, device_ids=device_ids)
            train_z = pre_encoder(X_train_cat).detach().cpu().numpy()
            np.save(os.path.join(vae_path, f'train_z_{train_target}.npy'), train_z)
            test_z = pre_encoder(X_test_cat).detach().cpu().numpy()
            np.save(os.path.join(vae_path, f'test_z_{train_target}.npy'), test_z)
            np.save(os.path.join(vae_path, f'train_num_prcsd.npy'), X_train_num.cpu().numpy())
            np.save(os.path.join(vae_path, f'test_num_prcsd.npy'), X_test_num.cpu().numpy())
