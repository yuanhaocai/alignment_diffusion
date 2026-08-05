import json
# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import os
import time
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

from ..utils_train import compute_norms
from ..utils_tuning import metric_dict_write
from .model_pm import MLPDiffusion_pm, Model_pm
from .data_pm import PredictiveModelingDataset


def _atomic_torch_save(payload, path):
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _save_training_checkpoint(
        checkpoint_path, model, optimizer, scheduler, epoch, best_loss, patience,
        log_history, time_used_total, best_model_epoch=None
    ):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "patience": patience,
        "log_history": log_history,
        "time_used_total": time_used_total,
        "best_model_epoch": best_model_epoch,
    }
    _atomic_torch_save(checkpoint, checkpoint_path)


def _normalize_milestone_epochs(save_milestone_epochs):
    if save_milestone_epochs is None or save_milestone_epochs == "":
        return set()
    if isinstance(save_milestone_epochs, str):
        values = save_milestone_epochs.replace(";", ",").split(",")
    else:
        values = save_milestone_epochs
    return {int(v) for v in values if str(v).strip()}


def diffusion_train_pm(
        y_emb_path, emb_alignment_path, diffusion_path, 
        y_type: Literal['continuous', 'categorical'], 
        use_tab: bool = True, use_text: bool = True, use_image: bool = False,
        vae_dat_path=None, text_path=None, image_path=None,
        epochs=5000, batch_size=4096, lr=1e-4, d_in=-1, dim_t=1024, 
        cond_dim=None,
        text_pooling="mean",
        normalize_tab_cond=True,
        normalize_text_tokens=False,
        normalize_text_pooled=True,
        normalize_projected_cond=False,
        alignment_mix_alpha=1.0, raw_vae_dat_path=None, raw_text_path=None,
        early_stopping_thrshd=500, clip_grad=False, clip_grad_max_norm=2.0,
        diffusion_class_weights=None, normalize_diffusion_class_weights=True,
        save_milestone_epochs=None,
        use_wandb=False, wandb_project=None, run_group=None, run_name=None, wandb_log_interval=10,
        resume=False, checkpoint_path=None, checkpoint_interval=1, save_checkpoint=True,
        seed=None,
        **kwargs
    ):
    """
    Train the diffusion model for predictive modeling, built upon ``diffusion_train``.

    The difference here is that the diffusion process is on the response, and the diffusion model is
    conditioned on combinations of predictor, text, and image.

    Removed all transfer learning, dp, and ddp implementations.

    seed: if not None, seed Python/NumPy/torch (+CUDA) and the DataLoader generator for
    reproducible training. Default None preserves the original (unseeded) behaviour.
    """
    assert use_tab or use_text or use_image
    device = "cuda"

    loader_generator = None
    if seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
        print(f"[seed] training seeded with seed={seed}", flush=True)

    if not os.path.exists(diffusion_path):
        os.makedirs(diffusion_path)
    if cond_dim is None:
        cond_dim = 768
    if checkpoint_path is None:
        checkpoint_path = os.path.join(diffusion_path, 'checkpoint.pt')
    training_complete_path = os.path.join(diffusion_path, 'training_complete.json')
    milestone_epochs = _normalize_milestone_epochs(save_milestone_epochs)
    diffusion_class_weights = (
        None if diffusion_class_weights is None
        else [float(v) for v in diffusion_class_weights]
    )

    diff_hypars = {
        'y_emb_path': y_emb_path,
        'emb_alignment_path': emb_alignment_path,
        'diffusion_path': diffusion_path,
        'y_type': y_type,
        'use_tab': use_tab,
        'use_text': use_text,
        'use_image': use_image,
        'vae_dat_path': vae_dat_path,
        'text_path': text_path,
        'image_path': image_path,
        'd_in': d_in,
        'dim_t': dim_t,
        'cond_dim': cond_dim,
        'text_pooling': text_pooling,
        'normalize_tab_cond': normalize_tab_cond,
        'normalize_text_tokens': normalize_text_tokens,
        'normalize_text_pooled': normalize_text_pooled,
        'normalize_projected_cond': normalize_projected_cond,
        'alignment_mix_alpha': alignment_mix_alpha,
        'raw_vae_dat_path': raw_vae_dat_path,
        'raw_text_path': raw_text_path,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'early_stopping_thrshd': early_stopping_thrshd,
        'clip_grad': clip_grad,
        'clip_grad_max_norm': clip_grad_max_norm,
        'diffusion_class_weights': diffusion_class_weights,
        'normalize_diffusion_class_weights': normalize_diffusion_class_weights,
        'save_milestone_epochs': sorted(milestone_epochs),
        'use_wandb': use_wandb,
        'wandb_project': wandb_project,
        'run_group': run_group,
        'run_name': run_name,
        'wandb_log_interval': wandb_log_interval,
        'resume': resume,
        'checkpoint_path': checkpoint_path,
        'checkpoint_interval': checkpoint_interval,
        'save_checkpoint': save_checkpoint,
    }
    with open(os.path.join(diffusion_path, 'hyperparameters.txt'), 'w') as file:
        metric_dict_write(file, diff_hypars, None)

    if wandb_log_interval is None:
        use_wandb = False
        wandb_log_interval = 99999

    if wandb is None:
        if use_wandb:
            raise ImportError("wandb is required when use_wandb=True.")
    else:
        wandb.init(
            project=wandb_project, 
            group=run_group, 
            name=run_name,
            config=diff_hypars, 
            mode='online' if use_wandb else 'disabled',
            job_type='train',
        )

    train_dataset = PredictiveModelingDataset(
        y_emb_path, emb_alignment_path, y_type, 'train',
        vae_dat_path, text_path, image_path, 
        use_tab, use_text, use_image,
        alignment_mix_alpha=alignment_mix_alpha,
        raw_vae_dat_path=raw_vae_dat_path,
        raw_text_path=raw_text_path,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        generator=loader_generator,
    )

    print("use_tab:", use_tab, "\n")
    print("use_text:", use_text, "\n")
    print("use_image:", use_image, "\n")
    print("text_pooling:", text_pooling, "\n")
    print("normalize_tab_cond:", normalize_tab_cond, "\n")
    print("normalize_text_tokens:", normalize_text_tokens, "\n")
    print("normalize_text_pooled:", normalize_text_pooled, "\n")
    print("normalize_projected_cond:", normalize_projected_cond, "\n")
    denoise_fn = MLPDiffusion_pm(
        d_in, dim_t, cond_dim=cond_dim, 
        use_tab=use_tab, use_text=use_text, use_image=use_image,
        text_pooling=text_pooling,
        normalize_tab_cond=normalize_tab_cond,
        normalize_text_tokens=normalize_text_tokens,
        normalize_text_pooled=normalize_text_pooled,
        normalize_projected_cond=normalize_projected_cond,
    ).to(device)
    model = Model_pm(
        denoise_fn=denoise_fn,
        class_weights=diffusion_class_weights,
        normalize_class_weights=normalize_diffusion_class_weights,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad is True)
    print("The number of trainable parameters:", num_params)

    optimizer = torch.optim.Adam(model.denoise_fn_D.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    # wandb.watch(model.denoise_fn_D, log='all', log_freq=1)  # significantly slow down model training.

    model.train()
    best_loss = float('inf')
    best_model_epoch = None
    patience = 0
    log_history = []
    start_epoch = 0
    time_used_so_far = 0.0

    if resume:
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_loss = checkpoint.get("best_loss", best_loss)
            best_model_epoch = checkpoint.get("best_model_epoch", best_model_epoch)
            patience = checkpoint.get("patience", patience)
            log_history = checkpoint.get("log_history", log_history)
            start_epoch = checkpoint.get("epoch", -1) + 1
            time_used_so_far = checkpoint.get("time_used_total", 0.0)
            print(f"Resumed at epoch {start_epoch + 1}/{epochs}")
        else:
            print(f"resume=True, but checkpoint not found at {checkpoint_path}. Starting from scratch.")

    start_time = time.time()
    last_epoch = start_epoch - 1
    stop_training = False

    if start_epoch >= epochs:
        print(f"Checkpoint already reached requested epochs: {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")

        log_dict = {}
        batch_loss = 0.0
        len_input = 0

        for y_emb, tab_pred_emb, text_emb, img_emb in pbar:
            y_emb = y_emb.float().to(device)
            tab_pred_emb, text_emb, img_emb = tab_pred_emb.float().to(device), text_emb.float().to(device), img_emb.float().to(device)
            loss = model(y_emb, tab_pred_emb, text_emb, img_emb)
            loss = loss.mean()
            batch_loss += loss.item() * len(y_emb)
            len_input += len(y_emb)

            optimizer.zero_grad()
            loss.backward()

            if clip_grad:
                grad_norm_, param_norm_ = compute_norms(list(model.denoise_fn_D.parameters()))
                log_dict.update({
                    'grad_norm_unclipped': grad_norm_,
                })
                torch.nn.utils.clip_grad_norm_(model.denoise_fn_D.parameters(), max_norm=clip_grad_max_norm)

            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        # save
        epoch_number = epoch + 1
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_model_epoch = epoch_number
            patience = 0
            torch.save(model.state_dict(), os.path.join(diffusion_path, 'model.pt'))
        else:
            patience += 1
            if patience == early_stopping_thrshd:
                print('Early stopping')
                stop_training = True
        if epoch_number in milestone_epochs:
            torch.save(model.state_dict(), os.path.join(diffusion_path, f'model_epoch{epoch_number}.pt'))
        if (epoch + 1) % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(diffusion_path, f'model_{epoch}.pt'))
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {curr_loss:.6f}, Best Loss: {best_loss:.6f}, Patience: {patience}")

        # log most infos, some logged in the loop
        grad_norm, param_norm = compute_norms(list(model.denoise_fn_D.parameters()))
        log_dict.update({
            'epoch': epoch,
            'loss': curr_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'patience': patience,
            'grad_norm': grad_norm,
            'param_norm': param_norm,
        })
        log_history.append(log_dict)

        if wandb is not None and (epoch + 1) % wandb_log_interval == 0:
            wandb.log(log_dict, step=epoch)

        if save_checkpoint:
            should_checkpoint = (
                (checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0)
                or stop_training
                or (epoch + 1) == epochs
            )
            if should_checkpoint:
                _save_training_checkpoint(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_loss=best_loss,
                    patience=patience,
                    log_history=log_history,
                    time_used_total=time_used_so_far + (time.time() - start_time),
                    best_model_epoch=best_model_epoch,
                )

        if stop_training:
            break

    end_time = time.time()
    time_used_total = time_used_so_far + (end_time - start_time)
    print('Time: ', end_time - start_time)
    print('Total time including previous resumed sessions: ', time_used_total)

    log_history = pd.DataFrame(
        log_history, 
        columns=['epoch', 'loss', 'lr', 'patience', 'grad_norm', 'param_norm', 'grad_norm_unclipped']
    )
    log_history.to_csv(os.path.join(diffusion_path, 'log_history.csv'), index=False)
    with open(os.path.join(diffusion_path, 'hyperparameters.txt'), 'a') as file:
        file.write(f"Epochs run: {last_epoch + 1}\n") 
        file.write(f"Time used: {time_used_total}\n")
    with open(training_complete_path, 'w') as file:
        json.dump(
            {
                "completed": True,
                "epochs_requested": epochs,
                "epochs_run": last_epoch + 1,
                "best_loss": best_loss,
                "best_model_epoch": best_model_epoch,
                "patience": patience,
                "time_used": time_used_total,
                "checkpoint_path": checkpoint_path,
            },
            file,
            indent=2,
        )
    if wandb is not None:
        wandb.finish()
